from egomimic.algo.act import TestModel

config_path = './mimicplay/configs/act.json'

test_model = TestModel(config_path)

test_model.run_test()
