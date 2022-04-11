import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch

gpu_id = 1
torch.cuda.set_device(gpu_id)

dataset = "FB15k-RETA"
model_name = "distmult"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/{}/".format(dataset),
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/{}/".format(dataset), "link", type_constrain=False)  # sampling_mode = 'link'

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
distmult.save_checkpoint('./checkpoint/{}/{}/distmult.ckpt'.format(dataset, model_name))
distmult.save_parameters('./checkpoint/{}/{}/distmult_emb.vec'.format(dataset, model_name))

# test the model
distmult.load_checkpoint('./checkpoint/{}/{}/distmult.ckpt'.format(dataset, model_name))
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
print("-----------------------test_link_prediction---------------------------\n")
tester.run_link_prediction(type_constrain = False)
print("-----------------------test_triple_classification---------------------------\n")
acc, threshlod = tester.run_triple_classification()
print("Accuracy: " + str(acc))
print("Threshold: " + str(threshlod))
