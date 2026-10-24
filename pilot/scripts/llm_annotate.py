# LLM seed annotation (EXTENDED_PROTOCOL.md section 2) with a local llama.cpp server.
# Starts llama-server, sends one grammar-constrained multiple-choice prompt per item, writes results incrementally
# (resumable), then stops the server.
# usage: python -I llm_annotate.py <items.jsonl> <out.jsonl> [--device cpu|gpu] [--limit N] [--threads 16]
import argparse, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

LLAMA = os.environ.get('NIRA_LLAMA_SERVER', 'llama-server')
MODEL = os.environ.get('NIRA_LLM_MODEL', 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
LETTERS = 'ABCDEFGHIJ'
LANG = {'zh_en': 'Chinese', 'ja_en': 'Japanese', 'fr_en': 'French'}

ap = argparse.ArgumentParser()
ap.add_argument('items'); ap.add_argument('out')
ap.add_argument('--device', default='gpu', choices=['cpu', 'gpu']); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--threads', type=int, default=16); ap.add_argument('--parallel', type=int, default=4)
ap.add_argument('--port', type=int, default=8091)
ap.add_argument('--server-exe', default=LLAMA)
ap.add_argument('--model', default=MODEL)
a = ap.parse_args()
LLAMA, MODEL = a.server_exe, a.model
URL = f'http://127.0.0.1:{a.port}'


def prompt(it):
    opts = '\n'.join(f'{LETTERS[i]}. {n}' for i, n in enumerate(it['cand_names']))
    if it.get('dataset') == 'FBDB15K':  # FBDB15K_PROTOCOL.md section 3
        user = ('Task: entity alignment between Freebase and DBpedia. Pick the DBpedia entity that refers to the same '
                'real-world entity as the source entity. Answer with the letter of the best option only.\n\n'
                f'Source entity (Freebase): {it["src_name"]}\n\nOptions:\n{opts}')
    else:
        user = ('Task: entity alignment between Wikipedia language editions. Pick the English Wikipedia entity that '
                'refers to the same real-world entity as the source entity. Answer with the letter of the best option '
                f'only.\n\nSource entity ({LANG[it["split"]]} Wikipedia): {it["src_name"]}\n\nOptions:\n{opts}')
    return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'


def post(path, body, timeout=120):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(it):
    r = post('/completion', {'prompt': prompt(it), 'n_predict': 1, 'temperature': 0.0, 'top_k': 1,
                             'grammar': 'root ::= [A-J]', 'cache_prompt': True, 'n_probs': 10})
    letter = r.get('content', '').strip()[:1]
    k = LETTERS.find(letter)
    prob = None
    try:
        prob = float(r['completion_probabilities'][0]['top_logprobs'][0]['logprob'])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    choice = it['cands'][k] if k >= 0 else None
    return dict(part=it['part'], src=it['src'], ref_index=it['ref_index'], gold=it['gold'], letter=letter,
                choice=choice, correct=choice == it['gold'], gold_in_cands=it['gold_in_cands'], top_logprob=prob)


items = [json.loads(l) for l in open(a.items, encoding='utf-8')]
done = set()
if os.path.exists(a.out):
    for l in open(a.out, encoding='utf-8'):
        r = json.loads(l); done.add((r['part'], r['src']))
todo = [it for it in items if (it['part'], it['src']) not in done]
if a.limit:
    todo = todo[:a.limit]
print(f'{len(done)} already done, {len(todo)} to annotate', flush=True)
if not todo:
    if not a.limit:
        open(a.out + '.done', 'w').write(f'{len(done)} items\n')
    sys.exit(0)

dev = ['--device', 'none', '-ngl', '0'] if a.device == 'cpu' else ['-ngl', '99']
server = subprocess.Popen([LLAMA, '-m', MODEL, '--port', str(a.port), '-c', str(1024 * a.parallel), '-np', str(a.parallel),
                           '-t', str(a.threads), '--no-webui'] + dev,
                          stdout=open(a.out + '.server.log', 'w'), stderr=subprocess.STDOUT)
try:
    for _ in range(600):  # /health is a GET endpoint; 200 once the model is loaded
        try:
            if urllib.request.urlopen(URL + '/health', timeout=5).status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit('llama-server did not become healthy')
    t0, n = time.time(), 0
    with open(a.out, 'a', encoding='utf-8') as f, ThreadPoolExecutor(a.parallel) as ex:
        for res in ex.map(ask, todo):
            f.write(json.dumps(res) + '\n'); n += 1
            if n % 200 == 0:
                f.flush()
                print(f'{n}/{len(todo)} done, {n / (time.time() - t0):.2f} items/s', flush=True)
    print(f'finished {n} items in {(time.time() - t0) / 60:.1f} min ({n / (time.time() - t0):.2f} items/s)', flush=True)
    keys = {(json.loads(l)['part'], json.loads(l)['src']) for l in open(a.out, encoding='utf-8')}
    if not a.limit and all((it['part'], it['src']) in keys for it in items):
        open(a.out + '.done', 'w').write(f'{len(keys)} items\n')  # completion marker for the queue
finally:
    server.terminate()
    try:
        server.wait(30)
    except subprocess.TimeoutExpired:
        server.kill()
