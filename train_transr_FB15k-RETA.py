import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch

gpu_id = 1
torch.cuda.set_device(gpu_id)

dataset = "FB15k-RETA"
model_name = "transR"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/{}/".format(dataset),
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/{}/".format(dataset),
	sampling_mode = 'link',
	type_constrain=False)

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200,
	dim_r = 200,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = True)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("./result/transr_transe.json")

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transr.save_checkpoint('./checkpoint/{}/{}/transr.ckpt'.format(dataset, model_name))
transr.save_parameters('./checkpoint/{}/{}/transr_emb.vec'.format(dataset, model_name))

# test the model
transr.load_checkpoint('./checkpoint/{}/{}/transr.ckpt'.format(dataset, model_name))
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True)
print("-----------------------test_link_prediction---------------------------\n")
tester.run_link_prediction(type_constrain = False)
print("-----------------------test_triple_classification---------------------------\n")
acc, threshlod = tester.run_triple_classification()
print("Accuracy: " + str(acc))
print("Threshold: " + str(threshlod))
