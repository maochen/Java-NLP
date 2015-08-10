package org.maochen.nlp.parser.stanford;

import com.google.common.collect.ImmutableSet;

import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangLib;
import org.maochen.nlp.parser.IParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import edu.stanford.nlp.ie.NERClassifierCombiner;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

/**
 * Created by Maochen on 7/20/15.
 */
public abstract class StanfordParser implements IParser {

    private static final Logger LOG = LoggerFactory.getLogger(StanfordParser.class);

    private static MaxentTagger posTagger = null;

    private static String posTaggerModelPath = null;

    private List<NERClassifierCombiner> ners;

    // This is for Lemma Tagger
    private static final Set<String> particles = ImmutableSet.of(
            "abroad", "across", "after", "ahead", "along", "aside", "away", "around",
            "back", "down", "forward", "in", "off", "on", "over", "out",
            "round", "together", "through", "up"
    );

    // 1. Tokenize
    public static List<CoreLabel> stanfordTokenize(String str) {
        TokenizerFactory<? extends HasWord> tf = PTBTokenizer.coreLabelFactory();

        // ptb3Escaping=false -> '(' not converted as '-LRB-', Dont use it, it will cause Dependency resolution err.
        Tokenizer<? extends HasWord> originalWordTokenizer = tf.getTokenizer(new StringReader(str), "ptb3Escaping=false");
        Tokenizer<? extends HasWord> tokenizer = tf.getTokenizer(new StringReader(str));

        List<? extends HasWord> originalTokens = originalWordTokenizer.tokenize();
        List<? extends HasWord> tokens = tokenizer.tokenize();
        // Curse you Stanford!
        List<CoreLabel> coreLabels = new ArrayList<>(tokens.size());

        for (int i = 0; i < tokens.size(); i++) {
            CoreLabel coreLabel = new CoreLabel();
            coreLabel.setWord(tokens.get(i).word());
            coreLabel.setOriginalText(originalTokens.get(i).word());
            coreLabel.setValue(tokens.get(i).word());
            coreLabels.add(coreLabel);
        }

        return coreLabels;
    }

    // 2. POS Tagger
    public void tagPOS(List<CoreLabel> tokens) {
        if (posTagger == null) {
            if (posTaggerModelPath == null) {
                LOG.warn("Default POS Tagger model");
                posTaggerModelPath = "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";
            }
            posTagger = new MaxentTagger(posTaggerModelPath);
        }
        List<TaggedWord> posList = posTagger.tagSentence(tokens);
        for (int i = 0; i < tokens.size(); i++) {
            String pos = posList.get(i).tag();
            tokens.get(i).setTag(pos);
        }
    }


    // For Lemma
    private static String phrasalVerb(Morphology morpha, String word, String tag) {
        // must be a verb and contain an underscore
        assert (word != null);
        assert (tag != null);
        if (!tag.startsWith(LangLib.POS_VB) || !word.contains("_")) return null;

        // check whether the last part is a particle
        String[] verb = word.split("_");
        if (verb.length != 2) return null;
        String particle = verb[1];
        if (particles.contains(particle)) {
            String base = verb[0];
            String lemma = morpha.lemma(base, tag);
            return lemma + '_' + particle;
        }

        return null;
    }

    // 3. Lemma Tagger
    public static void tagLemma(List<CoreLabel> tokens) {
        // Not sure if this can be static.
        Morphology morpha = new Morphology();

        for (CoreLabel token : tokens) {
            String lemma;
            String pos = token.tag();
            if (pos.equals(LangLib.POS_NNPS)) {
                pos = LangLib.POS_NNS;
            }
            if (pos.length() > 0) {
                String phrasalVerb = phrasalVerb(morpha, token.word(), pos);
                if (phrasalVerb == null) {
                    lemma = morpha.lemma(token.word(), pos);
                } else {
                    lemma = phrasalVerb;
                }
            } else {
                lemma = morpha.stem(token.word());
            }

            // LGLibEn.convertUnI only accept cap I.
            if (lemma.equals("i")) {
                lemma = "I";
            }

            token.setLemma(lemma);
        }
    }

    // 4. NER
    public void tagNamedEntity(List<CoreLabel> tokens) {
        boolean isPOSTagged = tokens.parallelStream().filter(x -> x.tag() == null).count() == 0;
        if (!isPOSTagged) {
            throw new RuntimeException("Please Run POS Tagger before Named Entity tagger.");
        }
        if (ners != null) {
            ners.stream().forEach(ner -> ner.classify(tokens));
        }
    }

    @Override
    public abstract DTree parse(final String sentence);

    protected void load(final String posTaggerModel, List<String> nerModels) {
        posTaggerModelPath = posTaggerModel;

        if (nerModels != null) {
            if (nerModels.isEmpty()) {
                nerModels.add("edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
                nerModels.add("edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz");
            }

            // STUPID NER, Throw IOException in the constructor ... : (
            ners = nerModels.stream().map(path -> {
                NERClassifierCombiner nerClassifierCombiner = null;
                try {
                    nerClassifierCombiner = new NERClassifierCombiner(path);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                return nerClassifierCombiner;
            }).collect(Collectors.toList());
        }
    }
}
