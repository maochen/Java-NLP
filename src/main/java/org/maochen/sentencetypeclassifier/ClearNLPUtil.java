package org.maochen.sentencetypeclassifier;

import com.clearnlp.component.AbstractComponent;
import com.clearnlp.dependency.DEPTree;
import com.clearnlp.nlp.NLPGetter;
import com.clearnlp.nlp.NLPMode;
import com.clearnlp.reader.AbstractReader;
import com.clearnlp.tokenization.AbstractTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class ClearNLPUtil {
    private static final Logger LOG = LoggerFactory.getLogger(ClearNLPUtil.class);

    final String language = AbstractReader.LANG_EN;
    final String modelType = "general-en";

    AbstractTokenizer tokenizer;
    AbstractComponent tagger;
    AbstractComponent parser;
    AbstractComponent identifier;
    AbstractComponent classifier;
    AbstractComponent labeler;

    List<AbstractComponent> components;

    public ClearNLPUtil() {
        try {
            long start = System.currentTimeMillis();
            LOG.info("Loading ClearNLP Models ...");
            tokenizer = NLPGetter.getTokenizer(language);
            tagger = NLPGetter.getComponent(modelType, language, NLPMode.MODE_POS);
            parser = NLPGetter.getComponent(modelType, language, NLPMode.MODE_DEP);
            identifier = NLPGetter.getComponent(modelType, language, NLPMode.MODE_PRED);
            classifier = NLPGetter.getComponent(modelType, language, NLPMode.MODE_ROLE);
            labeler = NLPGetter.getComponent(modelType, language, NLPMode.MODE_SRL);

            components = new ArrayList<AbstractComponent>();
            components.add(tagger);
            components.add(parser);
            components.add(identifier);
            components.add(classifier);
            components.add(labeler);
            long end = System.currentTimeMillis();
            long duration = (end - start) / 1000;
            LOG.info("Loading completed ... " + duration + " secs.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public DEPTree process(String sentence) {
        DEPTree tree = NLPGetter.toDEPTree(tokenizer.getTokens(sentence));

        for (AbstractComponent component : components)
            component.process(tree);

        return tree;
    }


}