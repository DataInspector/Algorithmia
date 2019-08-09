# Copyright 2019 H2O.ai; Proprietary License;  -*- encoding: utf-8 -*-

import sys

sys.path.append('gen-py')

from h2oai_scoring import ScoringService
from h2oai_scoring.ttypes import Row
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from tornado.options import define, options, parse_command_line
import pandas as pd
from numpy import nan
from numpy import ma
from scoring_h2oai_experiment_4e86cfd8_ba96_11e9_b4c6_0242ac110002 import Scorer

scoring_module_hash = '4e86cfd8-ba96-11e9-b4c6-0242ac110002'

def _convert(a):
    return a.item() if isinstance(a, np.generic) else a


class TCPHandler(object):
    def __init__(self):
        self.scorer = Scorer()

    def get_hash(self):
        return scoring_module_hash

    def score(self, row, output_margin=False, pred_contribs=False):
        return self.scorer.score([
            row.sepalLen if hasattr(row, 'sepalLen') else None,  # sepal_len
            row.sepalWid if hasattr(row, 'sepalWid') else None,  # sepal_wid
            row.petalLen if hasattr(row, 'petalLen') else None,  # petal_len
            row.petalWid if hasattr(row, 'petalWid') else None,  # petal_wid
        ], output_margin, pred_contribs)

    def score_batch(self, rows, output_margin=False, pred_contribs=False):
        columns = [
            pd.Series([r.sepalLen if hasattr(r, 'sepalLen') else None for r in rows], name='sepal_len', dtype='float32'),
            pd.Series([r.sepalWid if hasattr(r, 'sepalWid') else None for r in rows], name='sepal_wid', dtype='float32'),
            pd.Series([r.petalLen if hasattr(r, 'petalLen') else None for r in rows], name='petal_len', dtype='float32'),
            pd.Series([r.petalWid if hasattr(r, 'petalWid') else None for r in rows], name='petal_wid', dtype='float32'),
        ]
        pr = self.scorer.score_batch(pd.concat(columns, axis=1), output_margin, pred_contribs).values
        return pr.tolist()

    def get_target_labels(self):
        labels = self.scorer.get_target_labels()
        if labels is None:
            return []
        else:
            return labels.astype(str)

    def get_transformed_column_names(self):
        return self.scorer.get_transformed_column_names()

    def get_column_names(self):
        return self.scorer.get_column_names()

    def get_prediction_column_names(self, output_margin=False, pred_contribs=False):
        return self.scorer.get_pred_columns(output_margin, pred_contribs)

def start_tcp_server(port):
    scoring_handler = TCPHandler()
    server = TServer.TForkingServer(
        ScoringService.Processor(scoring_handler),
        TSocket.TServerSocket(port=port),
        TTransport.TBufferedTransportFactory(),
        TBinaryProtocol.TBinaryProtocolFactory(),
    )
    print('TCP scoring service listening on port {}...'.format(port))
    server.serve()


define('port', default=9090, help='Port to run scoring server on.', type=int)

def main():
    parse_command_line()
    start_tcp_server(options.port)


if __name__ == "__main__":
    main()

