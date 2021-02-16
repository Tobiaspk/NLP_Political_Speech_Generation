import torch
from torch.utils.data import Dataset
class GPT2Dataset(Dataset):
  """
    Dataset wrapper for data specific
  """
  def __init__(self, txt_list, tokenizer, max_length=768):

    self.tokenizer = tokenizer
    self.max_length=max_length
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:
      encodings_dict = tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


####################

import os
import pickle5
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader

class GetData():
  def __init__(self, path, token_length=768):
    self.path = path
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<|endoftext|>")
    self.connect_drive()
    self.speeches = self.read_data()
    self.token_length = token_length

    # after run once, keep as batch
    self.dataset = None
    self.dataloader = None

  def connect_drive(self):
    """
      Connect to Google Drive Data if drive in path
    """
    if "/content/drive" in self.path:
      try:
          from google.colab import drive
          COLAB = True
          drive.mount('/content/drive')
      except:
        COLAB = False

  def read_data(self):
    """
      Read data as pandas dataframe
    """
    path_temp = os.path.join(self.path, "Possible_datasets", "president_speeches.pkl")
    with open(path_temp, "rb") as f:
      speeches = pickle5.load(f)
    print(f"Shape: {', '.join([str(i) for i in speeches.shape])}")
    return speeches

  def get_data(self):
    """
      Return the dataset
    """
    return self.speeches

  def get_dataset(self, speech_list: list = None):
    """
      Create pytorch dataset
    """
    if speech_list is None:
      speech_list = self.get_data().speech.tolist()

    self.dataset = GPT2Dataset(
        txt_list=speech_list,
        tokenizer=self.tokenizer,
        max_length=self.token_length
    )
    
    return self.dataset

  def get_dataloader(self, dataset: Dataset = None, batch_size=2):
    from torch.utils.data import RandomSampler

    if dataset is None:
      if self.dataset is None:
        dataset = self.get_dataset()
      else:
        dataset = self.dataset

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset), # Select batches randomly
        batch_size=batch_size # Trains with this batch size.
    )
    self.dataloader = dataloader
    return dataloader

  def get_data_of(self,
                  president_name: str = None,
                  after_year: int = None):
    data_temp = self.get_data()
    if president_name is not None:
      data_temp = data_temp[data_temp.president_name == president_name]
    if after_year is not None:
      data_temp = data_temp[data_temp.year >= after_year]
    return data_temp

  def get_dataset_of(self, *args, **kwargs):
    data_temp = self.get_data_of(**kwargs)
    dataset_temp = self.get_dataset(speech_list=data_temp.speech.tolist())
    return dataset_temp

  def get_dataloader_of(self,
                        president_name: str = None,
                        after_year: int = None,
                        batch_size=2):
    """
      Return dataloader based on a subset of the data
    """
    dataset_temp = self.get_dataset_of(president_name=president_name,
                                       after_year=after_year)
    dataloader_temp = self.get_dataloader(dataset=dataset_temp, batch_size=batch_size)
    return dataloader_temp

  def list_presidents(self, top=5):
    print(self.get_data().president_name.value_counts().head(top))



#####################


import time
import datetime
from copy import deepcopy
import torch
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW

def format_time(elapsed):
  return str(datetime.timedelta(seconds=int(round((elapsed)))))

class PresidentModel():
  def __init__(self, model_input: GPT2LMHeadModel = None):
    self.device = torch.device("cuda")
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<|endoftext|>")
    self.model = self.init_model(model_input=model_input)
    self.model.to(self.device)

  def init_model(self, model_input: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """
      Initialise model if model is given as parameter
    """
    if model_input is None:
      return GPT2LMHeadModel.from_pretrained("gpt2",
                                             pad_token_id=self.tokenizer.eos_token_id)
    else:
      return model_input

  def fine_tune(self,
                data,
                epochs: int = 1,
                learning_rate: float = 5e-5,
                epsilon: float = 1e-8,
                warmup_steps: int = 100) -> None:
    # check input
    assert type(data) == DataLoader, "Datatype for 'data' must be"
    "DataLoader"

    # define implicite variables
    batch_size = data.batch_size
    total_steps = len(data) * epochs
    sample_every = 20

    # define optimizer
    optimizer = AdamW(self.model.parameters(),
                      lr=learning_rate,
                      eps=epsilon)

    # define scheduler for learningrate strategy
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps)

    self.model.train()  # put model in training mode (for dropout etc.)
    for epoch_i in range(0, epochs):
        t0 = time.time()
        for step, batch in enumerate(data):
            # set batch values
            b_input_ids = batch[0].to(self.device)
            b_labels = batch[0].to(self.device)
            b_masks = batch[1].to(self.device) 
            self.model.zero_grad() # reset gradients to not accumulate! 

            # forward propagation
            outputs = self.model.forward( 
                b_input_ids,
                labels=b_labels, 
                #attention_mask = b_masks,
                token_type_ids=None)

            # update params
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_loss = loss.item()

            # print in-between times
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(data), round(batch_loss, 4), elapsed))


  def generate_sentences(self,
                         prompt: str = "",
                         num_sentences: int = 3,
                         max_length: int = 20,
                         num_beams: int = 50,
                         no_repeat_ngram_size: int = 3,
                         print_it: bool = True,
                         cuda: bool = True):
    if type(prompt) == str:
      prompt = [prompt]
    input_ids = self.tokenizer(prompt,
                                return_tensors='pt',
                                padding=True,
                                truncation=True)["input_ids"]

    input_ids = input_ids.to(self.device)
    output_temp = self.model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=num_beams, 
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        top_k=0,
        num_return_sequences=num_sentences
    )
    if print_it:
      for i in range(num_sentences):
        print(self.tokenizer.decode(output_temp[i]))

    self.model = self.model.to(self.device)
    return output_temp

  def copy_model(self):
    return deepcopy(self.model)

  def copy(self):
    return deepcopy(self)
