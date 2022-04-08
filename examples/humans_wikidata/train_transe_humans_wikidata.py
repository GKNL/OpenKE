import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

dataset = "humans_wikidata"
model_name = "transE"

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
test_dataloader = TestDataLoader("./benchmarks/{}/".format(dataset), "link", type_constrain=False)  # sampling_mode = 'link'

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),  # entTotal
	rel_tot = train_dataloader.get_rel_tot(),  # relTotal
	dim = 200,  # Embedding维度为200
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/{}/{}/transe.ckpt'.format(dataset, model_name))
transe.save_parameters('./checkpoint/{}/{}/transe_emb.vec'.format(dataset, model_name))

# test the model
transe.load_checkpoint('./checkpoint/{}/{}/transe.ckpt'.format(dataset, model_name))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
print("-----------------------test_link_prediction---------------------------\n")
tester.run_link_prediction(type_constrain = False)
print("-----------------------test_triple_classification---------------------------\n")
acc, threshlod = tester.run_triple_classification()
print("Accuracy: " + str(acc))
print("Threshold: " + str(threshlod))
