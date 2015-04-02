package org.maochen.parser;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import edu.stanford.nlp.ie.NERClassifierCombiner;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.common.ParserQuery;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.SemanticHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.ScoredObject;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DNode;
import org.maochen.datastructure.DTree;
import org.maochen.datastructure.LangLib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.StringReader;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 12/8/14.
 */
public class StanfordParser implements IParser {

    private static final Logger LOG = LoggerFactory.getLogger(StanfordParser.class);

    private LexicalizedParser parser = null;

    private List<NERClassifierCombiner> ners = new ArrayList<>();

//    private static MaxentTagger posTagger = new MaxentTagger(System.getProperty("pos.model", MaxentTagger.DEFAULT_JAR_PATH));

    // This is for Lemma Tagger
    private static final Set<String> particles = ImmutableSet.of(
            "abroad", "across", "after", "ahead", "along", "aside", "away", "around",
            "back", "down", "forward", "in", "off", "on", "over", "out",
            "round", "together", "through", "up"
    );

    // 1. Tokenize
    private List<CoreLabel> stanfordTokenize(String str) {
        TokenizerFactory<? extends HasWord> tf = parser.getOp().tlpParams.treebankLanguagePack().getTokenizerFactory();
        // ptb3Escaping=false -> '(' not converted as '-LRB-', Dont use it, it will cause Dependency resolution err.
        // Tokenizer<? extends HasWord> tokenizer = tf.getTokenizer(new StringReader(str), "ptb3Escaping=false");
        Tokenizer<? extends HasWord> tokenizer = tf.getTokenizer(new StringReader(str));
        return (List<CoreLabel>) tokenizer.tokenize();
    }

    // 2. Correct Specific Input
    private void tagForm(List<CoreLabel> tokens) {
        Map<String, String> specialChar = new HashMap<String, String>() {
            {
                put("-LSB-", "[");
                put("-RSB-", "]");
                put("-LRB-", "(");
                put("-RRB-", ")");
                put("-LCB-", "{");
                put("-RCB-", "}");
                put("``", "\"");
                put("''", "\"");
            }
        };

        tokens.parallelStream().filter(token -> specialChar.containsKey(token.word())).forEach(token -> {
            String text = specialChar.get(token.word());
            token.setWord(text);
            token.setOriginalText(text);
        });
    }

    // 3. POS Tagger
    private void tagPOS(List<CoreLabel> tokens, Tree tree) {
        List<TaggedWord> posList = tree.getChild(0).taggedYield();
        for (int i = 0; i < tokens.size(); i++) {
            String word = tokens.get(i).word();
            String pos = posList.get(i).tag();
            switch (word) {
                case "(":
                    pos = "-LRB-";
                    break;
                case ")":
                    pos = "-RRB-";
                    break;
                case "-":
                    pos = "HYPH";
                    break;
            }
            tokens.get(i).setTag(pos);
        }
    }

    // 4. Lemma Tagger
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

    // 5. NER
    private void tagNamedEntity(List<CoreLabel> tokens) {
        ners.stream().forEach(ner -> ner.classify(tokens));
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

    /**
     * This is a piece of mystery code. It allows copula as head!!! Dont touch this unless you have full confidence.
     * This code cannot be found in their Javadoc....
     * By Maochen
     */
    // What is Mary happy about? -- copula
    private Collection<TypedDependency> getDependencies(Tree tree, boolean makeCopulaVerbHead) {
        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head
        // string -> true return all tokens including punctuations.
        Collection<TypedDependency> typedDependencies = new EnglishGrammaticalStructure(tree, string -> true, headFinder).typedDependencies();
        return typedDependencies;
    }

    public void loadModel(String modelFileLoc) {
        if (!modelFileLoc.isEmpty()) {
            parser = LexicalizedParser.loadModel(modelFileLoc, new ArrayList<>());
        }
    }

    public LexicalizedParser getLexicalizedParser() {
        return parser;
    }

    @Override
    public DTree parse(String sentence) {
        if (parser == null) {
            LOG.info("Use default PCFG model.");
            parser = LexicalizedParser.loadModel();
        }

        List<CoreLabel> tokens = stanfordTokenize(sentence);
        // Parse right after get through tokenizer.
        Tree tree = parser.parse(tokens);
        Collection<TypedDependency> dependencies = getDependencies(tree, true);

        tagForm(tokens);
        tagPOS(tokens, tree);
        tagLemma(tokens);
        tagNamedEntity(tokens);

        DTree depTree = StanfordTreeBuilder.generate(tokens, tree, dependencies);
        return depTree;
    }


    public Table<DTree, Tree, Double> getKBestParse(String sentence, int k) {
        if (parser == null) {
            LOG.info("Use default PCFG model.");
            parser = LexicalizedParser.loadModel();
        }

        List<CoreLabel> tokens = stanfordTokenize(sentence);

        // Parse right after get through tokenizer.
        ParserQuery pq = parser.parserQuery();
        pq.parse(tokens);
        List<ScoredObject<Tree>> scoredTrees = pq.getKBestPCFGParses(k);

        tagForm(tokens);
        tagNamedEntity(tokens);

        Table<DTree, Tree, Double> result = HashBasedTable.create();
        for (ScoredObject<Tree> scoredTuple : scoredTrees) {
            Tree tree = scoredTuple.object();
            tagPOS(tokens, tree);
            tagLemma(tokens);

            Collection<TypedDependency> dependencies = getDependencies(tree, true);
            DTree depTree = StanfordTreeBuilder.generate(tokens, tree, dependencies);
            result.put(depTree, tree, scoredTuple.score());
        }
        return result;
    }

    public List<String> tokenize(String sentence) {
        List<CoreLabel> tokens = stanfordTokenize(sentence);
        tagForm(tokens);
        return tokens.stream().parallel().map(CoreLabel::originalText).collect(Collectors.toList());
    }

    public StanfordParser() {
        this(StringUtils.EMPTY, true);
    }

    public StanfordParser(String modelPath, boolean initNER) {
        loadModel(modelPath);

        if (initNER) {
            // STUPID NER, Throw IOException in the constructor ... : (
            try {
                ners.add(new NERClassifierCombiner("edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz"));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    private static String print(boolean useForRecordErr, DTree tree) {
        if (useForRecordErr) {
            StringBuilder builder = new StringBuilder();

            for (DNode node : tree) {
                if (node.equals(tree.getPaddingNode())) {
                    continue;
                }
                builder.append(node.getId()).append(StringUtils.SPACE);
                builder.append(node.getForm()).append(StringUtils.SPACE);
                builder.append(node.getLemma()).append(StringUtils.SPACE);
                builder.append(node.getPOS()).append(StringUtils.SPACE);
                builder.append(node.getHead().getId()).append(StringUtils.SPACE);
                builder.append(node.getDepLabel());
                builder.append(System.lineSeparator());
            }

            return builder.toString();
        } else {
            return tree.toString();
        }
    }

    public static void main(String[] args) {
        StanfordParser parser = new StanfordParser("", false);

        parser.loadModel("/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/englishPCFG.ser.gz");
        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;

        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
//                                System.out.println(print(false, parser.parse(input)));


                Table<DTree, Tree, Double> trees = parser.getKBestParse(input, 3);

                List<Table.Cell<DTree, Tree, Double>> results = trees.cellSet().parallelStream().collect(Collectors.toList());
                Collections.sort(results, (o1, o2) -> Double.compare(o2.getValue(), o1.getValue()));
                for (Table.Cell<DTree, Tree, Double> entry : results) {
                    System.out.println("--------------------------");
                    System.out.println(entry.getValue());
                    System.out.println(entry.getColumnKey().pennString());
                    System.out.println("");
                    System.out.println(print(false, entry.getRowKey()));
                }


            }
        }

    }
}
