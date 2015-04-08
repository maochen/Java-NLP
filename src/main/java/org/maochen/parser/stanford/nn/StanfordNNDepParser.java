package org.maochen.parser.stanford.nn;

import com.google.common.collect.ImmutableSet;
import edu.stanford.nlp.ie.NERClassifierCombiner;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import org.maochen.datastructure.DTree;
import org.maochen.datastructure.LangLib;
import org.maochen.parser.IParser;
import org.maochen.parser.StanfordTreeBuilder;

import java.io.FileNotFoundException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordNNDepParser implements IParser {

    private static final TokenizerFactory<Word> tf = PTBTokenizer.PTBTokenizerFactory.newTokenizerFactory();
    private static final MaxentTagger posTagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
    public static DependencyParser nndepParser = null;

    private List<NERClassifierCombiner> ners = new ArrayList<>();

    // This is for Lemma Tagger
    private static final Set<String> particles = ImmutableSet.of(
            "abroad", "across", "after", "ahead", "along", "aside", "away", "around",
            "back", "down", "forward", "in", "off", "on", "over", "out",
            "round", "together", "through", "up"
    );

    // 1. Tokenize
    private List<CoreLabel> stanfordTokenize(String str) {
        Tokenizer<Word> originalWordTokenizer = tf.getTokenizer(new StringReader(str), "ptb3Escaping=false");
        Tokenizer<Word> tokenizer = tf.getTokenizer(new StringReader(str));

        List<Word> originalTokens = originalWordTokenizer.tokenize();
        List<Word> tokens = tokenizer.tokenize();
        // Curse you Stanford!
        List<CoreLabel> coreLabels = new ArrayList<>(tokens.size());

        for (int i = 0; i < tokens.size(); i++) {
            CoreLabel coreLabel = new CoreLabel();
            coreLabel.setWord(tokens.get(i).word());
            coreLabel.setOriginalText(originalTokens.get(i).word());
            coreLabels.add(coreLabel);
        }

        return coreLabels;
    }

    // 2. POS Tagger
    private void tagPOS(List<CoreLabel> tokenizedSentence) {
        List<TaggedWord> tokens = posTagger.tagSentence(tokenizedSentence);

        for (int i = 0; i < tokens.size(); i++) {
            String pos = tokens.get(i).tag();
            tokenizedSentence.get(i).setTag(pos);
        }
    }

    // 3. Lemma Tagger
    private void tagLemma(List<CoreLabel> tokens) {
        // Not sure if this can be static.
        Morphology morpha = new Morphology();

        for (CoreLabel token : tokens) {
            String lemma;
            if (token.tag().length() > 0) {
                String phrasalVerb = phrasalVerb(morpha, token.word(), token.tag());
                if (phrasalVerb == null) {
                    lemma = morpha.lemma(token.word(), token.tag());
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

    // For Lemma
    private String phrasalVerb(Morphology morpha, String word, String tag) {
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

    // 4. Dependency Label
    private GrammaticalStructure tagDependencies(List<? extends HasWord> taggedWords) {
        GrammaticalStructure gs = nndepParser.predict(taggedWords);
        return gs;
    }

    public String getCoNLLXString(GrammaticalStructure gs, Tree tree) {
        String conllx = EnglishGrammaticalStructure.dependenciesToString(gs, gs.typedDependencies(), tree, true, true);
        return conllx;
    }

    // 5. Named Entity Tagger
    private void tagNamedEntity(List<CoreLabel> tokens) {
        ners.stream().forEach(ner -> ner.classify(tokens));
    }

    public StanfordNNDepParser() {
        this(null, true);
    }

    public StanfordNNDepParser(final String inputModelPath, boolean initNER) {
        String modelPath = inputModelPath == null || inputModelPath.trim().isEmpty() ? DependencyParser.DEFAULT_MODEL : inputModelPath;
        nndepParser = DependencyParser.loadFromModelFile(modelPath);

        if (initNER) {
            // STUPID NER, Throw IOException in the constructor ... : (
            try {
                ners.add(new NERClassifierCombiner("edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz"));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public DTree parse(String sentence) {
        List<CoreLabel> tokenizedSentence = stanfordTokenize(sentence);
        tagPOS(tokenizedSentence);
        tagLemma(tokenizedSentence);
        Collection<TypedDependency> dependencies = tagDependencies(tokenizedSentence).typedDependencies();
        tagNamedEntity(tokenizedSentence);
        DTree depTree = StanfordTreeBuilder.generate(tokenizedSentence, dependencies, null);
        return depTree;
    }

    // This is just for debugging.
    public GrammaticalStructure getGrammaticalStructure(String sentence) {
        List<CoreLabel> tokenizedSentence = stanfordTokenize(sentence);
        tagPOS(tokenizedSentence);
        tagLemma(tokenizedSentence);
        return tagDependencies(tokenizedSentence);
    }

    public static void main(String[] args) {
        StanfordNNDepParser parser = new StanfordNNDepParser(DependencyParser.DEFAULT_MODEL, false);
        String text = "Mary can almost (always) tell when movies use fake dinosaurs and make changes.";
        System.out.println(parser.parse(text));
    }
}
