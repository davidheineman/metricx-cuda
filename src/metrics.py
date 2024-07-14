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

        device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False)

        self.metric = MT5ForMetricX.from_pretrained(model_id)
        self.metric.to(device)
        self.metric.eval()

    def prepare_inputs(self, src, pred, ref, max_input_length=1024):
        """Custom function for creating a MetricX dataset class using a lists for input."""
        def _preprocess(example):
            # add prompt
            if self.requires_references:
                example["input"] = f'candidate: {example["hypothesis"]} reference: {example["reference"]}'
            else:
                example["input"] = f'candidate: {example["hypothesis"]} source: {example["source"]}'

            # tokenize
            example = self.tokenizer(
                example["input"],
                max_length=max_input_length,
                truncation=True,
                padding='max_length',
            )

            # remove EOS
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]

            return example
        
        data = []
        if self.requires_references:
            assert ref is not None
            for s, p, r in zip(src, pred, ref):
                if isinstance(r, list): 
                    assert len(r) == 1, f"MetricX only supports a single reference! Recieved: {r}"
                    r = r[0]
                data += [{'source': s, 'hypothesis': p, 'reference': r}]
        else:
            for s, p in zip(src, pred):
                data += [{ 'source': s, 'hypothesis': p }]

        ds = Dataset.from_list(data)
        ds = ds.map(_preprocess)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.metric.device,
            output_all_columns=True,
        )
        return ds['input_ids'], ds['attention_mask']
        
    def __call__(self, src, pred, ref=None):
        input_ids, attention_mask = self.prepare_inputs(src, pred, ref)
        
        evaluation = self.metric.forward(input_ids, attention_mask)

        return evaluation.tolist()
        