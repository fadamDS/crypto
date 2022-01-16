from src.utils import load_fold
from src.models import CryptoDART
from src.settings import (non_train_cols,
                          assets,
                          weights,
                          names,
                          dart_base_params)


def main():

    # Folds to use
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    data_head_dir = 'data/gresearch/processed/'
    exp_name = 'dart_base'

    target = ['Target']

    for fold_id in folds:

        model_save_dir = f'models/fold_{fold_id}/'
        result_save_path = f'experiments/{exp_name}/results_fold_{fold_id}.csv'

        print(f'Working on fold {fold_id}')
        print('-------------\n')
        print('\n')

        # Get path to folds data
        data_path = data_head_dir + 'fold_' + str(fold_id) + '/'

        # Load data
        train = load_fold(data_path + 'train/')
        test = load_fold(data_path + 'test/')

        features = [col for col in train.columns if col not in non_train_cols]

        # Init Dart Model
        cryptoDart = CryptoDART(assets=assets,
                                weights=weights,
                                names=names,
                                params=dart_base_params)

        # Make data
        cryptoDart.make_data(features=features,
                             target=target, test=test, train=train)

        # Train
        cryptoDart.train()

        # Test
        cryptoDart.run_full_test()

        # Store model and results
        cryptoDart.save_models(model_save_dir, base_name=exp_name)

        print(
            f'Done with fold {fold_id}, test  score {cryptoDart.test_score}')
        print('-------------\n')
        print('\n')

    print('Done With Experiment')


if __name__ == '__main__':
    main()
