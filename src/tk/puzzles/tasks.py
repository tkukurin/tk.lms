"""Based on GPT puzzles by Francois Fleuret <francois@fleuret.org>
"""
import os
import tqdm

import torch

from . import world, utils

######################################################################


def masked_inplace_autoregression(
    model,
    batch_size,
    input,
    ar_mask,
    deterministic_synthesis,
    forbidden_tokens=None,
    logit_biases=None,
    progress_bar_desc="autoregression",
    device=torch.device("cpu"),
):
    assert input.size() == ar_mask.size()

    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            total=(input.size(0) + batch_size - 1) // batch_size,
        )

    with torch.autograd.no_grad():
        t = model.training
        model.eval()

        for input, ar_mask in batches:
            model.masked_inplace_autoregression(
                input,
                ar_mask,
                deterministic_synthesis,
                forbidden_tokens,
                logit_biases,
            )

        model.train(t)


######################################################################


class Task:
    def batches(self, split="train", nb_to_use=-1, desc=None):
        pass

    def vocabulary_size(self):
        pass

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        pass


######################################################################


class World(Task):
    def save_image(self, input, result_dir, filename, logger):
        img = world.seq2img(input.to("cpu"), self.height, self.width)
        image_name = os.path.join(result_dir, filename)
        utils.save_image(img.float() / 255.0, image_name, nrow=6, padding=4)
        logger(f"wrote {image_name}")

    def make_ar_mask(self, input):
        b = torch.arange(input.size(1), device=input.device) > input.size(1) // 2
        return b.long()[None, :].expand_as(input)

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        result_dir=None,
        logger=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.height = 6
        self.width = 8

        self.train_input = world.generate_seq(
            nb_train_samples, height=self.height, width=self.width
        ).to(device)

        self.test_input = world.generate_seq(
            nb_test_samples, height=self.height, width=self.width
        ).to(device)

        self.nb_codes = max(
            self.train_input.max(), self.test_input.max()) + 1

        self.train_quizzes = []
        self.test_quizzes = []

        if result_dir is not None:
            self.save_image(
                self.train_input[:72], result_dir, f"world_train.png", logger
            )

    def batches(self, split="train", desc=None):
        assert split in {"train", "test"}
        if split == "train":
            input = self.train_input
            quizzes = self.train_quizzes
        else:
            input = self.test_input
            quizzes = self.test_quizzes

        if len(quizzes) > 0:
            quizzes = torch.cat(quizzes, dim=0)
            if quizzes.size(0) > input.size(0) // 2:
                i = torch.randperm(input.size(0))[: input.size(0) // 2]
                quizzes = quizzes[i]

            i = torch.randperm(input.size(0))[: input.size(0) - quizzes.size(0)]
            input = input[i]

            self.nb_batch_samples_world = input.size(0)
            self.nb_batch_samples_quizzes = quizzes.size(0)

            input = torch.cat([input, quizzes], dim=0)
        else:
            self.nb_batch_samples_world = input.size(0)
            self.nb_batch_samples_quizzes = 0

        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis, nmax=1000
    ):
        def compute_accuracy(input, logger=None):
            input = input[:nmax]
            ar_mask = self.make_ar_mask(input)
            result = input.clone() * (1 - ar_mask)

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            nb_total, nb_correct = (
                input.size(0),
                (input == result).long().min(dim=1).values.sum(),
            )

            return nb_total, nb_correct

        train_nb_total, train_nb_correct = compute_accuracy(self.train_input)

        logger(
            f"accuracy_train {n_epoch} nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
        )

        test_nb_total, test_nb_correct = compute_accuracy(self.test_input, logger)

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        main_test_accuracy = test_nb_correct / test_nb_total
        logger(f"main_test_accuracy {n_epoch} {main_test_accuracy}")

        ##############################

        input = self.test_input[:96]
        ar_mask = self.make_ar_mask(input)
        result = input.clone() * (1 - ar_mask)

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            progress_bar_desc=None,
            device=self.device,
        )

        self.save_image(
            result[:72],
            result_dir,
            f"world_prediction_{n_epoch:04d}_{model.id:02d}.png",
            logger,
        )

        return main_test_accuracy

    def renew_samples(self, nb, for_train=True):
        input = self.train_input if for_train else self.test_input
        nb = min(nb, input.size(0))
        input[:-nb] = input[nb:].clone()
        input[-nb:] = world.generate_seq(nb, height=self.height, width=self.width).to(
            self.device
        )

    def store_new_quizzes(self, new_quizzes, for_train=True):
        if for_train:
            self.train_quizzes.append(new_quizzes)
        else:
            self.test_quizzes.append(new_quizzes)

    def create_new_quizzes(
        self,
        n_epoch,
        result_dir,
        logger,
        nb,
        model,
        other_models,
    ):
        ###############################################################
        # Generate quizzes with model

        quizzes = torch.empty(
            nb, self.height * self.width * 2 + 1, device=self.device, dtype=torch.int64
        )
        ar_mask = torch.full(quizzes.size(), 1, device=self.device)

        masked_inplace_autoregression(
            model,
            self.batch_size,
            quizzes,
            ar_mask,
            deterministic_synthesis=False,
            progress_bar_desc="creating quizzes",
            device=self.device,
        )

        ###############################################################
        # Create the reverse quizzes

        l = self.height * self.width
        direction = quizzes[:, l : l + 1]
        direction = world.token_forward * (
            direction == world.token_backward
        ) + world.token_backward * (direction == world.token_forward)
        reverse_quizzes = torch.cat(
            [quizzes[:, l + 1 :], direction, quizzes[:, :l]], dim=1
        )

        ar_mask = self.make_ar_mask(quizzes)

        ###############################################################
        # Check how many of the other models can solve them in both
        # directions

        nb_correct = []

        for m in other_models:
            result = quizzes.clone()

            masked_inplace_autoregression(
                m,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis=True,
                progress_bar_desc="solving quizzes",
                device=self.device,
            )

            correct = (quizzes == result).long().min(dim=-1).values

            reverse_result = reverse_quizzes.clone()

            masked_inplace_autoregression(
                m,
                self.batch_size,
                reverse_result,
                ar_mask,
                deterministic_synthesis=True,
                progress_bar_desc="solving reversed quizzes",
                device=self.device,
            )

            reverse_correct = (
                (reverse_quizzes == reverse_result).long().min(dim=-1).values
            )

            nb_correct.append((correct * reverse_correct)[None, :])

        nb_correct = torch.cat(nb_correct, dim=0)

        filename = os.path.join(result_dir, "correct_{n_epoch:04d}.dat")
        with open(filename, "w") as f:
            for k in nb_correct:
                f.write(f"{k}\n")

        return quizzes, nb_correct.sum(dim=0)