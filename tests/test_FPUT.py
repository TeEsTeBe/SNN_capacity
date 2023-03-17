import unittest
import run_FPUT


class TestFPUT(unittest.TestCase):

    def test_run_FPUT(self):
        args_dict = dict(
            alpha=0.25,
            tau_relax=10,
            nbr_batches=100000,
            warmup_batches=10,
            init_epsilon=0.05,
            trial=0,
            discrete=False,
            force=True,
            in_dim=1,
            uniques=4,
            in_width=64,
            in_variance=1 / 3,
            osc=64,
            datapath='./data/FPUT/',
        )
        run_FPUT.main(args_dict)


if __name__ == "__main__":
    unittest.main()
