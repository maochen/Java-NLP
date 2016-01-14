package org.maochen.nlp.parser.stanford.util;

/**
 * Stanford Model Locations.
 *
 * Created by Maochen on 1/14/16.
 */
public class StanfordConst {

    private static final String STANFORD_PREFIX = "edu/stanford/nlp/models/";
    public static final String STANFORD_DEFAULT_POS_EN_MODEL = STANFORD_PREFIX + "pos-tagger/english-left3words/english-left3words-distsim.tagger";
    public static final String STANFORD_DEFAULT_NER_3CLASS_EN_MODEL = STANFORD_PREFIX + "ner/english.all.3class.distsim.crf.ser.gz";
    public static final String STANFORD_DEFAULT_NER_7CLASS_EN_MODEL = STANFORD_PREFIX + "ner/english.muc.7class.distsim.crf.ser.gz";
    public static final String STANFORD_DEFAULT_PCFG_EN_MODEL = STANFORD_PREFIX + "lexparser/englishPCFG.ser.gz";

}
