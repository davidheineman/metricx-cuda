import torch
import transformers
from datasets import Dataset
from metricx23.models import MT5ForRegression


PRETTY_METRIC_NAMES = {
    "comet": "COMET",
    "comet_20": "COMET",
    "comet_22": "COMET",
    "comet_kiwi": "COMET Kiwi",
    "comet_kiwi_22": "COMET Kiwi",
    "comet_kiwi_23": "COMET Kiwi",
    "comet_kiwi_xxl": "COMET Kiwi XXL",
    "xcomet": "XCOMET",
    "metricx": "MetricX ↓",
    "metricx_qe": "MetricX QE ↓",
}


class AbstractMetric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.devices = kwargs.get('devices', None)
        self.batch_size = kwargs.get('batch_size', 1)
        self.requires_references = kwargs.get('requires_references', False)
        
        if not kwargs.get('is_endpoint', False):
            print(f'Initalizing metric {self}...')

    def __call__(self, src, pred, ref):
        """
        All metric functions are in the form of (source, prediction, reference), although not all three
        fields may be used. 

        src:    Array of source texts
        pred:   Array of predicted texts
        ref:    Array of arrays of reference texts for each source text

        e.g.,
            sources=["About 95 species are currently accepted."]
            predictions=["About 95 you now get in."]
            references=[["About 95 species are currently known.", "About 95 species are now accepted.", "95 species are now accepted."]]

        Returns: System-level scores across an entire dataset
        """
        raise NotImplementedError()
    
    def __str__(self):
        return PRETTY_METRIC_NAMES.get(self.name, self.name)
    
    def __repr__(self):
        return f'{self.__str__()}()'
    
    def get_endpoint_name(self):
        return 'http://localhost:{port}/' + self.name + '_eval'


class MetricX(AbstractMetric):
    def __init__(self, variation='metricx', size='xl', batch_size=8, **kwargs):
        super().__init__(name=variation, batch_size=batch_size, requires_references=('qe' in variation), **kwargs)

        is_qe = (self.name == 'metricx_qe')

        model_id = None
        if is_qe:
            match size:
                case 'l': model_id = "google/metricx-23-qe-large-v2p0"
                case 'xl': model_id = "google/metricx-23-qe-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-qe-xxl-v2p0"
        else:
            match size:
                case 'l': model_id = "google/metricx-23-large-v2p0"
                case 'xl': model_id = "google/metricx-23-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-xxl-v2p0"
        if model_id is None: raise NotImplementedError(f'MetricX variation {self.name} at size {size} not supported!')

        device = torch.device("mps")

        per_device_batch_size = self.batch_size # batch_size // (len(devices) if isinstance(devices, list) else devices)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")

        model = MT5ForRegression.from_pretrained(model_id)
        model.to(device)
        model.eval()

        training_args = transformers.TrainingArguments(
            output_dir='/dev',
            per_device_eval_batch_size=per_device_batch_size,
            dataloader_pin_memory=False,
            use_mps_device=True
        )
        self.metric = transformers.Trainer(model=model, args=training_args)

        self.metric.model.to("mps")

    def prepare_inputs(self, data, is_qe, max_input_length=1024):
        """
        Custom function for creating a MetricX dataset class using a lists for input.
        """
        def _make_input(example):
            if is_qe:
                example["input"] = f'candidate: {example["hypothesis"]} source: {example["source"]}'
            else:
                example["input"] = f'candidate: {example["hypothesis"]} reference: {example["reference"]}'
            return example

        def _tokenize(example):
            tokenized = self.tokenizer(
                example["input"],
                max_length=max_input_length,
                truncation=True,
                padding='max_length',
            )
            return tokenized

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        ds = Dataset.from_list(data)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device="mps", # self.metric.model.device,
            output_all_columns=True,
        )
        return ds
    
    def _metricx(self, src, pred, ref):
        data = []
        for s, p, r in zip(src, pred, ref):
            if isinstance(r, list):
                r = r[0]
            data.append({'source': s, 'hypothesis': p, 'reference': r})
          
        ds = self.prepare_inputs(data, is_qe=False)
        evaluation, _, _ = self.metric.predict(test_dataset=ds)

        # Very important! Lower is better for MetricX, so we negate before returning. Note the 
        # model evaluation logic does not do this when reporting final results.
        evaluation = -evaluation

        return evaluation.tolist()

    def _metricx_qe(self, src, pred, ref=None):
        data = []
        for s, p in zip(src, pred):
            data += [{ 'source': s, 'hypothesis': p }]

        ds = self.prepare_inputs(data, is_qe=True)
        evaluation, _, _ = self.metric.predict(test_dataset=ds)

        # Very important! Lower is better for MetricX, so we negate before returning. Note the 
        # model evaluation logic does not do this when reporting final results.
        evaluation = -evaluation

        return evaluation.tolist()
    
    def __call__(self, src, pred, ref=None):
        if 'qe' in self.name:
            return self._metricx_qe(src=src, pred=pred, ref=None)
        else:
            assert ref is not None, 'MetricX requires references!'
            return self._metricx(src=src, pred=pred, ref=ref)
