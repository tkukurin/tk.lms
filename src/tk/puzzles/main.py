"""Based on GPT puzzles by Francois Fleuret <francois@fleuret.org>
"""
import math, sys, argparse, time, tqdm, os, datetime, warnings

import torch
from torch import nn
from torch.nn import functional as F

from tk.puzzles import gpt, tasks

######################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = torch.device("cpu")

######################################################################

parser = argparse.ArgumentParser(
    description="An implementation of GPT with cache.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--log_filename", type=str, default="train.log", help=" ")

parser.add_argument("--result_dir", type=str, default=None)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--max_percents_of_test_in_train", type=int, default=1)

########################################

parser.add_argument("--nb_epochs", type=int, default=10000)

parser.add_argument("--batch_size", type=int, default=None)

parser.add_argument("--physical_batch_size", type=int, default=None)

parser.add_argument("--nb_train_samples", type=int, default=None)

parser.add_argument("--nb_test_samples", type=int, default=None)

parser.add_argument("--learning_rate", type=float, default=1e-4)

########################################

parser.add_argument("--model", type=str, default=None)

parser.add_argument("--dim_model", type=int, default=None)

parser.add_argument("--dim_keys", type=int, default=None)

parser.add_argument("--dim_hidden", type=int, default=None)

parser.add_argument("--nb_heads", type=int, default=None)

parser.add_argument("--nb_blocks", type=int, default=None)

parser.add_argument("--dropout", type=float, default=0.1)

########################################

parser.add_argument("--deterministic_synthesis", action="store_true", default=False)

parser.add_argument("--nb_gpts", type=int, default=5)

parser.add_argument("--check", action="store_true", default=False)

######################################################################

args = parser.parse_args()

if args.result_dir is None:
    args.result_dir = f"results_culture"

######################################################################

default_args = {
    "model": "37M",
    "batch_size": 100,
    "nb_train_samples": 250000,
    "nb_test_samples": 10000,
}

for k, v in default_args.items():
    if getattr(args, k) is None:
        setattr(args, k, v)

######################################################################

default_model_args = {
    "17K": {
        "dim_model": 32,
        "dim_keys": 32,
        "dim_hidden": 32,
        "nb_heads": 2,
        "nb_blocks": 2,
    },
    "4M": {
        "dim_model": 256,
        "dim_keys": 32,
        "dim_hidden": 1024,
        "nb_heads": 4,
        "nb_blocks": 6,
    },
    "37M": {
        "dim_model": 512,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 12,
    },
    "122M": {
        "dim_model": 768,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 24,
    },
    "352M": {
        "dim_model": 1024,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 48,
    },
}

if args.model in default_model_args:
    for k, v in default_model_args[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
else:
    raise ValueError(f"Unknown model {args.model}")

######################################################################

try:
    os.mkdir(args.result_dir)
except FileExistsError:
    print(f"result directory {args.result_dir} already exists")
    exit(1)

log_file = open(os.path.join(args.result_dir, args.log_filename), "a")

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


log_string(f"argv {' '.join(sys.argv)}")

for n in vars(args):
    log_string(f"args.{n} {getattr(args, n)}")


######################################################################

if args.check:
    args.nb_train_samples = 2500
    args.nb_test_samples = 100

if args.physical_batch_size is None:
    args.physical_batch_size = args.batch_size
else:
    assert args.batch_size % args.physical_batch_size == 0

assert args.nb_train_samples % args.batch_size == 0
assert args.nb_test_samples % args.batch_size == 0

task = tasks.World(
    nb_train_samples=args.nb_train_samples,
    nb_test_samples=args.nb_test_samples,
    batch_size=args.physical_batch_size,
    result_dir=args.result_dir,
    logger=log_string,
    device=device,
)

######################################################################

log_string(f"device {device}")

vocabulary_size = task.vocabulary_size()

log_string(f"vocabulary_size {vocabulary_size}")

######################################################################

# Compute the entropy of the training tokens

token_count = 0
for input in task.batches(split="train", desc="train-entropy"):
    token_count += F.one_hot(input, num_classes=task.vocabulary_size()).sum((0, 1))
token_probas = token_count / token_count.sum()
entropy = -torch.xlogy(token_probas, token_probas).sum()
train_set_perplexity = math.exp(entropy)

######################################################################
# A bit of paranoia never hurts

if args.max_percents_of_test_in_train >= 0:

    def subsets_as_tuples(batches, cs):
        s = set()
        for batch in batches:
            for x in batch:
                s.add(tuple([v.item() for v in x]))
                if len(s) == cs:
                    yield s
                    s = set()
        yield s

    nb_test, nb_in_train = 0, 0
    for test_subset in subsets_as_tuples(
        task.batches(split="test", desc="test-check"), 25000
    ):
        in_train = set()
        for train_subset in subsets_as_tuples(
            task.batches(split="train", desc="train-check"), 25000
        ):
            in_train.update(test_subset.intersection(train_subset))
        nb_in_train += len(in_train)
        nb_test += len(test_subset)

    log_string(
        f"data_check {nb_in_train*100/nb_test:.02f}% ({nb_in_train}/{nb_test}) of test samples are in the train set"
    )

    assert (
        nb_in_train <= args.max_percents_of_test_in_train * nb_test / 100
    ), f"More than {args.max_percents_of_test_in_train}% of test samples are in the train set"

##############################


def one_epoch(model, task):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()

    nb_train_samples, acc_train_loss = 0, 0.0

    for input in task.batches(split="train"):
        input = input.to(device)

        if nb_train_samples % args.batch_size == 0:
            optimizer.zero_grad()

        output = model(gpt.BracketedSequence(input)).x
        loss = F.cross_entropy(output.transpose(1, 2), input)
        acc_train_loss += loss.item() * input.size(0)

        nb_train_samples += input.size(0)

        loss.backward()

        if nb_train_samples % args.batch_size == 0:
            optimizer.step()

    train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))

    log_string(f"train_perplexity {n_epoch} {train_perplexity}")


######################################################################


def run_tests(model, task, deterministic_synthesis):
    with torch.autograd.no_grad():
        model.eval()

        nb_test_samples, acc_test_loss = 0, 0.0
        nb_samples_accumulated = 0

        for input in task.batches(split="test"):
            input = input.to(device)

            bs = model(gpt.BracketedSequence(input))
            output = bs.x

            loss = F.cross_entropy(output.transpose(1, 2), input)

            acc_test_loss += loss.item() * input.size(0)

            nb_test_samples += input.size(0)

        main_test_accuracy = task.produce_results(
            n_epoch=n_epoch,
            model=model,
            result_dir=args.result_dir,
            logger=log_string,
            deterministic_synthesis=deterministic_synthesis,
        )

        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(f"test_perplexity {n_epoch} {test_perplexity}")

    model.main_test_accuracy = main_test_accuracy


######################################################################


def create_quizzes(
    model,
    other_models,
    task,
    nb_for_train=1000,
    nb_for_test=100,
):
    kept = []

    while sum([x.size(0) for x in kept]) < nb_for_train + nb_for_test:
        new_quizzes, nb_correct = task.create_new_quizzes(
            n_epoch=n_epoch,
            result_dir=args.result_dir,
            logger=log_string,
            nb=4 * (nb_for_train + nb_for_test),
            model=model,
            other_models=other_models,
        )

        print(nb_correct)

        to_keep = new_quizzes[nb_correct == len(other_models) - 1]
        log_string(f"keep {to_keep.size(0)} quizzes")
        kept.append(to_keep)

    new_quizzes = torch.cat(kept, dim=0)[: nb_for_train + nb_for_test]

    task.store_new_quizzes(new_quizzes[:nb_for_train], for_train=True)
    task.store_new_quizzes(new_quizzes[nb_for_train:], for_train=False)

    task.save_image(
        new_quizzes[:72],
        args.result_dir,
        f"world_quiz_{n_epoch:04d}_{model.id:02d}.png",
        log_string,
    )


######################################################################

models = []

for k in range(args.nb_gpts):
    model = gpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=args.dim_model,
        dim_keys=args.dim_keys,
        dim_hidden=args.dim_hidden,
        nb_heads=args.nb_heads,
        nb_blocks=args.nb_blocks,
        causal=True,
        dropout=args.dropout,
    ).to(device)

    model.main_test_accuracy = 0.0
    model.id = k

    models.append(model)


nb_parameters = sum(p.numel() for p in models[0].parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

######################################################################

accuracy_to_make_quizzes = 0.975
nb_new_quizzes_for_train = 1000
nb_new_quizzes_for_test = 100

if args.check:
    accuracy_to_make_quizzes = 0.0
    nb_new_quizzes_for_train = 10
    nb_new_quizzes_for_test = 10

for n_epoch in range(args.nb_epochs):
    a = [(model.id, float(model.main_test_accuracy)) for model in models]
    a.sort(key=lambda p: p[0])
    log_string(f"current accuracies {a}")

    # select the model with lowest accuracy
    models.sort(key=lambda model: model.main_test_accuracy)
    model = models[0]

    log_string(
        f"training model {model.id} main_test_accuracy {model.main_test_accuracy}"
    )

    # improve it
    one_epoch(model, task)

    task.renew_samples(args.nb_train_samples // args.nb_gpts)

    log_string(
        f"train_set_composition world {task.nb_batch_samples_world} quizzes {task.nb_batch_samples_quizzes}"
    )

    # test it
    run_tests(model, task, deterministic_synthesis=False)

    if min([m.main_test_accuracy for m in models]) >= accuracy_to_make_quizzes:
        other_models = models.copy()
        other_models.remove(model)

        create_quizzes(
            model,
            other_models,
            task,
            nb_for_train=nb_new_quizzes_for_train,
            nb_for_test=nb_new_quizzes_for_test,
        )

        # We update everyone
        for model in models:
            run_tests(model, task, deterministic_synthesis=False)


######################################################################