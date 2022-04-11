import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

dataset = "FB15k-RETA"
model_name = "rotate"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/{}/".format(dataset),
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/{}/".format(dataset), "link", type_constrain=False)

# define the model
rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	margin = 6.0,
	epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 3000, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
rotate.save_checkpoint('./checkpoint/{}/{}/rotate.ckpt'.format(dataset, model_name))
rotate.save_parameters('./checkpoint/{}/{}/rotate_emb.vec'.format(dataset, model_name))

# test the model
rotate.load_checkpoint('./checkpoint/{}/{}/rotate.ckpt'.format(dataset, model_name))
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
print("-----------------------test_link_prediction---------------------------\n")
tester.run_link_prediction(type_constrain = False)
print("-----------------------test_triple_classification---------------------------\n")
acc, threshlod = tester.run_triple_classification()
print("Accuracy: " + str(acc))
print("Threshold: " + str(threshlod))