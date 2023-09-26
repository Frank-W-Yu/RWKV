
from flask import Flask, request, jsonify
import argparse

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import numpy as np
from prompt_toolkit import prompt
try:
    # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
parser = argparse.ArgumentParser(description='Flask app with command-line arguments')
parser.add_argument('--model_name', type=str, help='Model path')
parser.add_argument('--strategy', type=str, help='Runtime strategy')

cml_args = parser.parse_args()
args = types.SimpleNamespace()

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

args.strategy = cml_args.strategy

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

app = Flask(__name__)
CHAT_LANG = 'Chinese' # English // Chinese // more to come
args.MODEL_NAME = cml_args.model_name


CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.2 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.5 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.4 # Presence Penalty
GEN_alpha_frequency = 0.4 # Frequency Penalty
GEN_penalty_decay = 0.996
AVOID_REPEAT = '，：？！'

CHUNK_LEN = 256 # split input into chunks to save VRAM (shorter -> slower)


if args.MODEL_NAME.endswith('/'): # for my own usage
    if 'rwkv-final.pth' in os.listdir(args.MODEL_NAME):
        args.MODEL_NAME = args.MODEL_NAME + 'rwkv-final.pth'
    else:
        latest_file = sorted([x for x in os.listdir(args.MODEL_NAME) if x.endswith('.pth')], key=lambda x: os.path.getctime(os.path.join(args.MODEL_NAME, x)))[-1]
        args.MODEL_NAME = args.MODEL_NAME + latest_file

########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

# Load Model

print(f'Loading model - {args.MODEL_NAME}')
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    END_OF_TEXT = 0
    END_OF_LINE = 11
else:
    print(f"Loading config {current_path}/20B_tokenizer.json")
    pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
    END_OF_TEXT = 0
    END_OF_LINE = 187
    END_OF_LINE_DOUBLE = 535
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None
AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

def run_rnn(tokens,model_tokens, model_state, newline_adj = 0):

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out, model_state

all_state = {}
def save_all_stat(srv, name, last_out, model_tokens, model_state):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
def fix_tokens(tokens):
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        return tokens
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens

########################################################################################################

# Run inference

# gc.collect()
torch.cuda.empty_cache()

def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def on_message(message):
    global model_tokens, model_state, user, bot, interface, init_prompt

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()
    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()
    print(f"Got message: {msg}")
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return
    
    # use '+prompt {path}' to load a new prompt
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            msg = msg[3:].strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN+100):
            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])
            
            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = pipeline.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25) # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            # if token == END_OF_TEXT:
            #     break
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay            
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            
            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = pipeline.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
            
            # send_msg = pipeline.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{pipeline.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

########################################################################################################
srv = "flask_server"

class NovelModel:
    def __init__(self) -> None:
        self.model_tokens = []
        self.model_state = None
    
    def reset(self)->None:
        self.model_tokens = []
        self.model_state = None


@app.route('/gen', methods=['POST'])
def gen():
    # schema {"message": "novel paragragh"}
    data = request.get_json()
    message = '\n'+data['message'].strip()
    print(message)
    model_state = None
    model_tokens = []
    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    out, model_state = run_rnn(pipeline.encode(message), model_tokens, model_state)
    save_all_stat(srv, 'gen_0', out, model_tokens, model_state)    
    begin = len(model_tokens)
    # print('model token:', begin)
    out_last = begin
    occurrence = {}
    ret = []
    for i in range(FREE_GEN_LEN+100):
        for n in occurrence:
            out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
        token = pipeline.sample_logits(
            out,
            temperature=x_temp,
            top_p=x_top_p,
        )
        if token == END_OF_TEXT:
            break
        for xxx in occurrence:
            occurrence[xxx] *= GEN_penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        out, model_state = run_rnn([token], model_tokens, model_state)
        
        xxx = pipeline.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx: # avoid utf-8 display issues
            # return jsonify({'payloads': xxx}), 200
            ret.append(xxx)
            print(xxx, end='', flush=True)
            out_last = begin + i + 1
            if i >= FREE_GEN_LEN:
                print("\nBefore break:", xxx)
                break
    # Return a JSON response
    print('final output: ', ''.join(ret))
    return jsonify({'payloads': ''.join(ret)}), 200

@app.route('/')
def hello_world():
    print("Flask server is running")
    return 'Flask server is working!'

print(f'{args.MODEL_NAME} - {args.strategy}')
if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)