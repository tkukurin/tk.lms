"""Dataset generator.

Moved from encoding.
"""
import json
from pathlib import Path
from absl import app, flags, logging
import datasets as hfd
import functools as ft

import tk
from tk.arc.encoding import (
    SepConfig, tokenize_with_sep, induce_vocab, encode,
    dslfunc2callable, constants_from_solve, problems,
    ProblemSpec, Io, IoW
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'out_dir',
    str(tk.datadir / "mhodel_rearc"),
    'Output directory')
flags.DEFINE_enum(
    'dataset_type', 
    'full', 
    ['full', 'prog_only'],
    'Dataset type to generate')
# flags.mark_flag_as_required('out_dir')
# flags.mark_flag_as_required('dataset_type')


def maybe_truncate_train_examples(
    spec, max_tokens: int, spec2toks
):
    out, types = spec2toks(spec)
    meta = {'skipped': []}
    while len(out) > max_tokens:
        if len(spec.train.i) == 0:
            logging.warning(f"{spec.id}: Can't truncate further ({len(out)=})")
            return None, meta
        meta['skipped'].append(IoW(spec.train.i[-1], spec.train.o[-1]))
        spec = ProblemSpec(
            spec.id,
            Io(
                spec.train.i[:-1],
                spec.train.o[:-1]
            ),
            spec.test,
            spec.func
        )
        out, types = spec2toks(spec)
    return (out, types), meta


def generate_full_dataset(problems, max_tokens=2048):
    vocab = set(dslfunc2callable) | set(constants_from_solve)
    problems_tokenized = {
        k: maybe_truncate_train_examples(
            spec,
            max_tokens=max_tokens, 
            spec2toks=ft.partial(
                tokenize_with_sep, 
                sep_config=SepConfig(),
                padlen=max_tokens,
                vocab=vocab
            )
        )
        for k, spec in problems.items()
    }
    
    vocab = induce_vocab(problems_tokenized)
    dataset = hfd.Dataset.from_dict({
        'id': list(problems_tokenized.keys()),
        'input_ids': [
            encode(toks, vocab)
            for (toks, _), _ in problems_tokenized.values()
            if len(toks) <= max_tokens
        ],
        'token_type_ids': [
            [t.value for t in types]
            for (_, types), _ in problems_tokenized.values()
            if len(types) <= max_tokens
        ]
    })
    return dataset, vocab, problems_tokenized

def generate_prog_only_dataset(problems_tokenized, vocab):
    only_programs_dataset = {}
    for k, ((toks, *_), *_) in problems_tokenized.items():
        toks_after_prog = toks[toks.index('<prog>'):toks.index('<end>')]
        only_programs_dataset[k] = {'input_ids': toks_after_prog}

    maxlen = max(len(x['input_ids']) for x in only_programs_dataset.values())
    for k, v in only_programs_dataset.items():
        v['input_ids'] = v['input_ids'] + [vocab['__config']['sep_pad']] * (maxlen - len(v['input_ids']))

    ds_programs = hfd.Dataset.from_dict({
        'id': [k for k in only_programs_dataset],
        'input_ids': [
            encode(v['input_ids'], vocab)
            for v in only_programs_dataset.values()],
    })
    return ds_programs

def save_dataset(dataset, vocab, outdir: Path):
    dataset.save_to_disk(outdir)
    with open(outdir / 'vocab.json', 'w') as f:
        json.dump(vocab, f)
    logging.info(f"Saved dataset to {outdir}")
    logging.info(f"Vocab size: {len(vocab)}")

def main(argv):
    outdir = Path(FLAGS.out_dir)
    dataset_type = FLAGS.dataset_type

    dataset, vocab, problems_tokenized = generate_full_dataset(problems)
    
    print(f'Saving dataset {dataset_type} to {outdir}')
    if dataset_type == 'full':
        save_dataset(
            dataset.with_format('jax'),
            vocab, 
            outdir
        )
    elif dataset_type == 'prog_only':
        ds_programs = generate_prog_only_dataset(problems_tokenized, vocab)
        save_dataset(
            ds_programs.with_format('jax'),
            vocab,
            outdir
        )

if __name__ == '__main__':
    print('running')
    app.run(main)
