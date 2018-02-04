#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))


    inp = '../policy.seg'
    outp1 = '../policy.model'
    outp2 = '../policy.vector'

    model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=5,
            workers=multiprocessing.cpu_count(), iter=3)

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    print('OK')

