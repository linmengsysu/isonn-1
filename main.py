import torch

from ntrainer import Trainer
from config import get_config

from data_loader import get_test_loader, get_train_valid_loader, load_dataset
import numpy as np
import time

def main(config):

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.cuda:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    scores = []
    # instantiate data loaders
    count = 0
    times =[]
    

    for i in [1,2,3]:
        start = time.time()
        count = i
        train_data, test_data = load_dataset(config.data_dir, str(count))
        # instantiate data loaders


        data_loader = get_train_valid_loader(
            train_data, config.batch_size,
            config.random_seed, config.valid_size,
            config.shuffle, config.show_sample, **kwargs
        )

        test_loader = get_test_loader(
            test_data, config.batch_size, **kwargs
        )


        # instantiate trainer
        trainer = Trainer(config, count, data_loader, test_loader)

        trainer.train()
        result = trainer.test()
        
        scores.append(result)
        elapsed = time.time() - start
        times.append(elapsed)

    scores = np.array(scores)
    times = np.array(times)
    print('>>> scores:',scores)
    print('aver time', times.mean())
    # print('avg\tacc\tf1\tprec\trec\tauc')
    print('acc:',scores.mean(axis=0)[0], '\nf1', scores.mean(axis=0)[1], '\nprec', scores.mean(axis=0)[2], '\nrec', scores.mean(axis=0)[3])
    print('>>> std')
    print('acc:',scores.std(axis=0)[0], '\nf1', scores.std(axis=0)[1], '\nprec', scores.std(axis=0)[2], '\nrec', scores.std(axis=0)[3])


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
