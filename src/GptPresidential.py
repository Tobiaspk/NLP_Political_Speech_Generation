import os
import time
import datetime
import pickle5
import torch
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW


# Note: not only GPT2

def format_time(elapsed):
  # print nicely formated elapsed time
  return str(datetime.timedelta(seconds=int(round((elapsed)))))

class GPT2Dataset(Dataset):
  """
    Pytorch Dataset wrapper that helps with training and batches of training data. Reads in texts of politicians
    :param txt_list: (Numpy) array of speeches.
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
class GetData():
  """
  Retrieve dataset from local or google colab. Helps generate pytorch dataloaders.
  :param data_path: path to president_speeches file
  :param token_length: Max token length to use in GPT2 model
  :param drive: Boolean if google drive should be mounted (used in google colab)
  """
  def __init__(self, data_path, token_length=768, drive=False):
    self.data_path = data_path
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<|endoftext|>")
    if drive:
      self.connect_drive() # used for working on colab
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
    with open(self.data_path, "rb") as f:
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
      Create a custom pytorch dataset specialised for this task. A customer speech list can be used as well.
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
    """
      Create an iterable dataloader based on a GPT2Dataset and specified batch_size. Attention, a large batchsize quickly leads
      to memory overloads.
      :param dataset: Input GPT2Dataset object
      :param batch_size: Integer of desired batch_size. Smaller equal 2 recommended.
    """
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
    """
      Easy filter wrapper of data. Specify the president and years of desired data.
      :param president_name: name of president
      :param after_year: choose only speeches of thisyear and later only
    """
    data_temp = self.get_data()
    if president_name is not None:
      data_temp = data_temp[data_temp.president_name == president_name]
    if after_year is not None:
      data_temp = data_temp[data_temp.year >= after_year]
    return data_temp

  def get_dataset_of(self, *args, **kwargs):
    """
      Returns a custom pytorch dataset (GPT2Dataset) using a subset of data. See get_data_of(...)
    """
    data_temp = self.get_data_of(**kwargs)
    dataset_temp = self.get_dataset(speech_list=data_temp.speech.tolist())
    return dataset_temp

  def get_dataloader_of(self,
                        president_name: str = None,
                        after_year: int = None,
                        batch_size=2):
    """
      Return dataloader based on a subset of the data. See get_data_of(...)
    """
    dataset_temp = self.get_dataset_of(president_name=president_name,
                                       after_year=after_year)
    dataloader_temp = self.get_dataloader(dataset=dataset_temp, batch_size=batch_size)
    return dataloader_temp

  def list_presidents(self, top=10):
    """
      List names of available presidents in dataset.
    """
    print(self.get_data().president_name.value_counts().head(top))



#####################
class PresidentModel():
  """
    Wrapper to easily train and finetune models based on president speech data.
    :param model_input: Initially empty. Can be used to introduce a pretrained model.
    :param device: Specify device to use
  """
  def __init__(self, model_input: GPT2LMHeadModel = None, device="cuda"):
    self.device = device
    self.device = torch.device(self.device)  # Sloppily use Cuda GPU. 
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<|endoftext|>")  # extract the gpt2 tokenizer
    self.model = self.init_model(model_input=model_input)  # initiate model
    self.model.to(self.device)  # send model to device

  def init_model(self, model_input: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """
      Load pretrained model or use input model
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
    """
      Simplify warm-up steps to finetune the model based on some data input.
      :param data: A dataloader. Easily retrieved using get_dataloader method of GetData class
      :param epochs: Integer, number of epochs
      :param learning_rate: Learning Rate used in Adam Optimizer
      :param epsilon: Epsilon used in Adam Optimizer
      :param warmup_steps: Number of warmup steps in a linear schedule
    """
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
                         prompt="",
                         num_sentences: int = 3,
                         max_length: int = 20,
                         num_beams: int = 50,
                         no_repeat_ngram_size: int = 3,
                         print_it: bool = True,
                         cuda: bool = True):
    """
      Easily generate sentences using the model (based on a prompt).
    """
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
    """
      Return a copy of the model
    """
    return deepcopy(self.model)

  def copy(self):
    return deepcopy(self)
