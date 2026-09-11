import main


def test_parse_args_enables_sample_mode():
    args = main.parse_args(["--stage", "train", "--sample"])

    assert args.stage == "train"
    assert args.sample is True


def test_train_stage_passes_sample_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "run_training", lambda config, *, sample: calls.append(sample))
    monkeypatch.setattr(main.Pipeline, "log_stage", lambda self, name: None)
    monkeypatch.setattr(main.Pipeline, "log_complete", lambda self: None)

    main.Pipeline(sample=True).run_single_stage("train")

    assert calls == [True]
