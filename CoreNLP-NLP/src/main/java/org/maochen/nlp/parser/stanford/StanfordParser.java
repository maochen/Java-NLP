package org.maochen.nlp.parser.stanford;

import com.google.common.collect.ImmutableSet;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangLib;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.util.StanfordConst;
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

    private static String POS_TAGGER_MODEL_PATH = null;

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
            coreLabel.setBeginPosition(((CoreLabel) tokens.get(i)).beginPosition());
            coreLabel.setEndPosition(((CoreLabel) tokens.get(i)).endPosition());
            coreLabels.add(coreLabel);
        }

        return coreLabels;
    }

    // 2. POS Tagger
    public void tagPOS(List<CoreLabel> tokens) {
        if (posTagger == null) {
            if (POS_TAGGER_MODEL_PATH == null) {
                LOG.warn("Default POS Tagger model");
                POS_TAGGER_MODEL_PATH = StanfordConst.STANFORD_DEFAULT_POS_EN_MODEL;
            }
            posTagger = new MaxentTagger(POS_TAGGER_MODEL_PATH);
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
        if (!tag.startsWith(LangLib.POS_VB) || !word.contains("_")) {
            return null;
        }

        // check whether the last part is a particle
        String[] verb = word.split("_");
        if (verb.length != 2) {
            return null;
        }
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
    // NER not thread safe ...
    public synchronized void tagNamedEntity(List<CoreLabel> tokens) {
        boolean isPOSTagged = tokens.parallelStream().filter(x -> x.tag() == null).count() == 0;
        if (!isPOSTagged) {
            throw new RuntimeException("Please Run POS Tagger before Named Entity tagger.");
        }
        if (ners != null) {
            try {
                ners.stream().forEach(ner -> ner.classify(tokens));
            } catch (Exception e) {
                /* edu.stanford.nlp.util.RuntimeInterruptedException: java.lang.InterruptedException
                 at edu.stanford.nlp.util.HashIndex.addToIndex(HashIndex.java:173) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher$BranchStates.newBid(SequenceMatcher.java:902) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher$MatchedStates.<init>(SequenceMatcher.java:1288) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.getStartStates(SequenceMatcher.java:709) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.findMatchStartBacktracking(SequenceMatcher.java:488) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.findMatchStart(SequenceMatcher.java:449) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.find(SequenceMatcher.java:341) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.findNextNonOverlapping(SequenceMatcher.java:365) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ling.tokensregex.SequenceMatcher.find(SequenceMatcher.java:437) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.NumberNormalizer.findNumbers(NumberNormalizer.java:452) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.NumberNormalizer.findAndMergeNumbers(NumberNormalizer.java:721) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.time.TimeExpressionExtractorImpl.extractTimeExpressions(TimeExpressionExtractorImpl.java:184) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.time.TimeExpressionExtractorImpl.extractTimeExpressions(TimeExpressionExtractorImpl.java:178) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.time.TimeExpressionExtractorImpl.extractTimeExpressionCoreMaps(TimeExpressionExtractorImpl.java:116) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.time.TimeExpressionExtractorImpl.extractTimeExpressionCoreMaps(TimeExpressionExtractorImpl.java:104) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.regexp.NumberSequenceClassifier.runSUTime(NumberSequenceClassifier.java:340) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.regexp.NumberSequenceClassifier.classifyWithSUTime(NumberSequenceClassifier.java:138) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.regexp.NumberSequenceClassifier.classifyWithGlobalInformation(NumberSequenceClassifier.java:101) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.NERClassifierCombiner.recognizeNumberSequences(NERClassifierCombiner.java:267) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.NERClassifierCombiner.classifyWithGlobalInformation(NERClassifierCombiner.java:231) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 at edu.stanford.nlp.ie.NERClassifierCombiner.classify(NERClassifierCombiner.java:218) ~[stanford-corenlp-3.5.2.jar:3.5.2]
                 */
                LOG.warn("NER Classifier err for: " + tokens.stream().map(CoreLabel::word).collect(Collectors.joining(StringUtils.SPACE)));
            }
        }
    }

    @Override
    public abstract DTree parse(final String sentence);

    protected void load(final String posTaggerModel, List<String> nerModels) {
        POS_TAGGER_MODEL_PATH = posTaggerModel;

        if (nerModels != null) {
            if (nerModels.isEmpty()) {
                nerModels.add(StanfordConst.STANFORD_DEFAULT_NER_3CLASS_EN_MODEL);
                nerModels.add(StanfordConst.STANFORD_DEFAULT_NER_7CLASS_EN_MODEL);
            }

            // STUPID NER, throw IOException in the constructor ... : (
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
