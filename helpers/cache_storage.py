import os
import errno
import glob

import torch


class CacheStorage():
    def __init__(self, encoding_model: str = None):
        if encoding_model:
            self.save_path = os.path.join('results', encoding_model)
            self.validate_folder(self.save_path)

            self.snapshot_prefix = os.path.join(self.save_path, 'snapshot')
            self.snapshot_template = '_acc_{:.4f}_loss_{:.6f}_iter_{}_model{}.pt'

            self.best_snapshot_prefix = os.path.join(
                self.save_path, 'best_snapshot')
            self.best_snapshot_template = '_devacc_{}_devloss_{}__iter_{}_model{}.pt'
            self.encoding_model = encoding_model

    def load_model_snapshot(self, path: str):
        if not path:
            return None
            
        model = torch.load(path, map_location=lambda storage, location: storage.cuda())
        return model

    def save_snapshot(
            self,
            model,
            iteration: int,
            train_accuracy: float,
            train_loss: float):

        snapshot_path = self.snapshot_prefix + \
            self.snapshot_template.format(
                train_accuracy, train_loss, iteration, '')
                
        snapshot_encoder_path = self.snapshot_prefix + \
            self.snapshot_template.format(
                train_accuracy, train_loss, iteration, '.enc')

        # save model
        torch.save(model, snapshot_path)
        torch.save(model.encoder.state_dict(), snapshot_encoder_path)

        # delete previous 'best_snapshot' files
        for f in glob.glob(self.snapshot_prefix + '*'):
            if f != snapshot_path and f != snapshot_encoder_path:
                os.remove(f)

    def save_best_snapshot(
            self,
            model,
            iteration: int,
            dev_accuracy: float,
            dev_loss: float):

        snapshot_path = self.best_snapshot_prefix + \
            self.best_snapshot_template.format(
                dev_accuracy, dev_loss, iteration, '')
                
        snapshot_encoder_path = self.best_snapshot_prefix + \
            self.best_snapshot_template.format(
                dev_accuracy, dev_loss, iteration, '.enc')

        # save model
        torch.save(model, snapshot_path)
        torch.save(model.encoder.state_dict(), snapshot_encoder_path)

        # delete previous 'best_snapshot' files
        for f in glob.glob(self.best_snapshot_prefix + '*'):
            if f != snapshot_path and f != snapshot_encoder_path:
                os.remove(f)

    def validate_folder(self, name):
        try:
            os.makedirs(name)
        except OSError as ex:
            if ex.errno == errno.EEXIST and os.path.isdir(name):
                # ignore existing directory
                pass
            else:
                # a different error happened
                raise