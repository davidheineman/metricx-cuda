import torch
from transformers import AutoTokenizer
from datasets import Dataset
from metricx23.models import MT5ForMetricX


PRETTY_METRIC_NAMES = {
    "metricx": "MetricX ↓",
    "metricx_qe": "MetricX QE ↓",
}


class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.devices = kwargs.get('devices', None)
        self.batch_size = kwargs.get('batch_size', 1)
        self.requires_references = kwargs.get('requires_references', False)
        
        if not kwargs.get('is_endpoint', False):
            print(f'Initalizing metric {self}...')
    
    def __str__(self): 
        return PRETTY_METRIC_NAMES.get(self.name, self.name)

    def __repr__(self): 
        return f'{self.__str__()}()'


class MetricX(Metric):
    def __init__(self, variation='metricx', size='xl', batch_size=8, **kwargs):
        super().__init__(name=variation, batch_size=batch_size, requires_references=('qe' not in variation), **kwargs)

        model_id = None
        if self.requires_references:
            match size:
                case 'l': model_id = "google/metricx-23-large-v2p0"
                case 'xl': model_id = "google/metricx-23-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-xxl-v2p0"
        else:
            match size:
                case 'l': model_id = "google/metricx-23-qe-large-v2p0"
                case 'xl': model_id = "google/metricx-23-qe-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-qe-xxl-v2p0"
        if model_id is None: raise NotImplementedError(f'MetricX variation {self.name} at size {size} not supported!')

        device = torch.device("mps")
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False, use_fast=True)

        self.metric = MT5ForMetricX.from_pretrained(model_id)
        self.metric.to(device)
        self.metric.eval()

    def prepare_inputs(self, src, pred, ref, max_input_length=1024):
        """Custom function for creating a MetricX dataset class using a lists for input."""
        # format src/pred/ref using prompt
        data = []
        if self.requires_references:
            assert ref is not None
            for s, p, r in zip(src, pred, ref):
                if isinstance(r, list): 
                    assert len(r) == 1, f"MetricX only supports a single reference! Recieved: {r}"
                    r = r[0]
                data += [f'candidate: {p} reference: {r}']
        else:
            for s, p in zip(src, pred):
                data += [f'candidate: {p} source: {s}']
        
        import time
        start = time.time()

        # tokenize
        ds = self.tokenizer(
            data,
            max_length=max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.metric.device)

        # remove EOS
        ds["input_ids"] = ds["input_ids"][:, :-1]
        ds["attention_mask"] = ds["attention_mask"][:, :-1]

        end = time.time()
        print(f"Dataset func time: {(end - start):.4f} seconds ({(len(src)/(end - start)):.2f} sent/s)")

        return ds['input_ids'], ds['attention_mask']
        
    def __call__(self, src, pred, ref=None):
        input_ids, attention_mask = self.prepare_inputs(src, pred, ref)

        import time
        start = time.time()

        # evaluation = self.metric.forward(input_ids, attention_mask)
        # evaluation = evaluation.tolist()

        bs = 1 # self.batch_size
        evaluation = []
        n_batches = (len(input_ids) + bs - 1) // bs
        for i in range(n_batches):
            batch_input_ids      = input_ids[i*bs:(i+1)*bs]
            batch_attention_mask = attention_mask[i*bs:(i+1)*bs]
            
            batch_evaluation = self.metric.forward(batch_input_ids, batch_attention_mask)
            
            evaluation += batch_evaluation.tolist()

        end = time.time()
        print(f"Model runtime: {(end - start):.4f} seconds ({(len(src)/(end - start)):.2f} sent/s)")

        return evaluation
        
    def prepare_inputs_slow(self, src, pred, ref, max_input_length=1024):
        """
        Original implementation of prepare_inputs(), but the hashing for 
        the map() function is prohibitively slow, where it may only make
        sense for tokenizing 100K+ examples simultaneously.
        """
        def _preprocess(batch):
            # tokenize
            batch = self.tokenizer(
                batch['prompt'],
                max_length=max_input_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(self.metric.device)

            # remove EOS
            batch["input_ids"] = batch["input_ids"][:, :-1]
            batch["attention_mask"] = batch["attention_mask"][:, :-1]

            return batch
        
        data = []
        if self.requires_references:
            assert ref is not None
            for s, p, r in zip(src, pred, ref):
                if isinstance(r, list): 
                    assert len(r) == 1, f"MetricX only supports a single reference! Recieved: {r}"
                    r = r[0]
                data += [{'prompt': f'candidate: {p} reference: {r}'}]
        else:
            for s, p in zip(src, pred):
                data += [{'prompt': f'candidate: {p} source: {s}'}]

        # tokenize in batches
        ds = Dataset.from_list(data)

        ds = ds.map(_preprocess, batched=True, batch_size=1000)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.metric.device,
            output_all_columns=True,
        )
