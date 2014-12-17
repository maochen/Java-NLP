package org.maochen.parser;

import com.google.common.collect.ImmutableSet;
import edu.stanford.nlp.ie.NERClassifierCombiner;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.SemanticHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DTree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.StringReader;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 12/8/14.
 */
public class StanfordParser implements IParser {

    private static final Logger LOG = LoggerFactory.getLogger(StanfordParser.class);

    private static LexicalizedParser parser = LexicalizedParser.loadModel();
    private static MaxentTagger posTagger = new MaxentTagger(System.getProperty("pos.model", MaxentTagger.DEFAULT_JAR_PATH));

    private static NERClassifierCombiner ner = null;

    // STUPID NER, Throw IOException in the constructor ... : (
    static {
        try {
            ner = new NERClassifierCombiner("edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    // This is for Lemma Tagger
    private static final Set<String> particles = ImmutableSet.of(
            "abroad", "across", "after", "ahead", "along", "aside", "away", "around",
            "back", "down", "forward", "in", "off", "on", "over", "out",
            "round", "together", "through", "up"
    );


    // 1. Tokenize
    private static List<CoreLabel> stanfordTokenize(String str) {
        TokenizerFactory<? extends HasWord> tf = parser.getOp().tlpParams.treebankLanguagePack().getTokenizerFactory();
        // ptb3Escaping=false -> '(' not converted as '-LRB-', Dont use it, it will cause Dependency resolution err.
        // Tokenizer<? extends HasWord> tokenizer = tf.getTokenizer(new StringReader(str), "ptb3Escaping=false");
        Tokenizer<? extends HasWord> tokenizer = tf.getTokenizer(new StringReader(str));
        return (List<CoreLabel>) tokenizer.tokenize();
    }

    // 2. Correct Specific Input
    private static void tagForm(List<CoreLabel> tokens) {
        Map<String, String> specialChar = new HashMap<String, String>() {
            {
                put("-LSB-", "[");
                put("-RSB-", "]");
                put("-LRB-", "(");
                put("-RRB-", ")");
                put("``", "\"");
                put("''", "\"");
            }
        };
        for (CoreLabel token : tokens) {
            if (specialChar.containsKey(token.word())) {
                String text = specialChar.get(token.word());
                token.setWord(text);
                token.setOriginalText(text);
            }
        }
    }

    // 3. POS Tagger
    private static void tagPOS(List<CoreLabel> tokens) {
        List<TaggedWord> posWords = posTagger.tagSentence(tokens, false);
        for (int i = 0; i < tokens.size(); i++) {
            String word = tokens.get(i).word();
            String pos = posWords.get(i).tag();
            if (word.equals("(")) {
                pos = "-LRB-";
            } else if (word.equals(")")) {
                pos = "-RRB-";
            } else if (word.equals("-")) {
                pos = "HYPH";
            }
            tokens.get(i).setTag(pos);
        }
    }

    // 4. Lemma Tagger
    private static void tagLemma(List<CoreLabel> tokens) {
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

    // 5. NER
    private static void tagNamedEntity(List<CoreLabel> tokens) {
        ner.classify(tokens);
    }

    // For Lemma
    private static String phrasalVerb(Morphology morpha, String word, String tag) {
        // must be a verb and contain an underscore
        assert (word != null);
        assert (tag != null);
        if (!tag.startsWith("VB") || !word.contains("_")) return null;

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

    /**
     * This is a piece of mystery code. It allows copula as head!!! Dont touch this unless you have full confidence.
     * This code cannot be found in their Javadoc....
     * By Maochen
     */
    // What is Mary happy about? -- copula
    private static Collection<TypedDependency> getDependencies(Tree tree, boolean makeCopulaVerbHead) {
        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head
        Predicate<String> puncFilter = parser.treebankLanguagePack().punctuationWordRejectFilter();
        Collection<TypedDependency> result = new EnglishGrammaticalStructure(tree, puncFilter, headFinder).typedDependencies();
        Predicate<String> puncAllowFilter = parser.treebankLanguagePack().punctuationWordAcceptFilter();
        result.addAll(new EnglishGrammaticalStructure(tree, puncAllowFilter, headFinder).typedDependencies());
        return result;
    }

    public void loadModel(String modelFileLoc) {
        if (!modelFileLoc.isEmpty()) {
            parser = LexicalizedParser.loadModel(modelFileLoc, StringUtils.EMPTY);
        }
    }

    @Override
    public DTree parse(String sentence) {
        List<CoreLabel> tokens = stanfordTokenize(sentence);
        // Parse right after get through tokenizer.
        Tree tree = parser.parse(tokens);
        Collection<TypedDependency> dependencies = getDependencies(tree, true);

        tagForm(tokens);
        tagPOS(tokens);
        tagLemma(tokens);
        tagNamedEntity(tokens);

        DTree depTree = StanfordTreeBuilder.generate(tokens, tree, dependencies);
        return depTree;
    }

    public List<String> tokenize(String sentence) {
        List<CoreLabel> tokens = stanfordTokenize(sentence);
        tagForm(tokens);
        return tokens.stream().parallel().map(CoreLabel::originalText).collect(Collectors.toList());
    }


    public static void main(String[] args) {
        IParser parser = new StanfordParser();

        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;
        //        input = "What is blended learning?";
        //        input = "Example isn't another way to teach, it is the only way to teach.";
        //        input = "John is singing.";
        //        input = "John is a teacher.";
        //        input = "John is at the store.";

        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                DTree tree = parser.parse(input);
                System.out.println(tree.toString());
            }
        }

    }
}
