package org.maochen.nlp.parser.stanford.util;

import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangTools;
import org.maochen.nlp.parser.stanford.StanfordParser;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.maochen.nlp.parser.stanford.pcfg.StanfordTreeBuilder;

import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.SemanticHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordParserUtils {

    public static List<String> tokenize(final String sentence) {
        if (sentence == null) {
            return null;
        }

        List<CoreLabel> coreLabels = StanfordParser.stanfordTokenize(sentence);
        return coreLabels.stream().map(CoreLabel::word).collect(Collectors.toList());
    }

    public static List<String> segmenter(final String blob) {
        if (blob == null) {
            return null;
        }

        TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer
                .factory(new CoreLabelTokenFactory(), "normalizeCurrency=false,ptb3Escaping=false");

        Tokenizer<CoreLabel> tokenizer = tokenizerFactory.getTokenizer(new StringReader(blob));
        List<CoreLabel> tokens = new ArrayList<>();
        while (tokenizer.hasNext()) {
            tokens.add(tokenizer.next());
        }

        List<List<CoreLabel>> sentences = new WordToSentenceProcessor<CoreLabel>().process(tokens);

        int end;
        int start = 0;
        List<String> sentenceList = new ArrayList<>();

        for (List<CoreLabel> sentence : sentences) {
            end = sentence.get(sentence.size() - 1).endPosition();
            sentenceList.add(blob.substring(start, end).trim());
            start = end;
        }
        return sentenceList;
    }

    public static DTree getDTreeFromCoreNLP(Collection<TypedDependency> deps, List<CoreLabel> tokens) {
        Map<Integer, TypedDependency> indexedDeps = new HashMap<>(deps.size());
        for (TypedDependency dep : deps) {
            indexedDeps.put(dep.dep().index(), dep);
        }

        if (tokens.get(0).lemma() == null) {
            StanfordNNDepParser.tagLemma(tokens);
        }

        DTree tree = new DTree();
        int idx = 1;
        for (CoreLabel token : tokens) {
            String word = token.originalText();
            String pos = token.tag();
            String cPos = (token.get(CoreAnnotations.CoarseTagAnnotation.class) != null) ? token.get(CoreAnnotations.CoarseTagAnnotation.class) : LangTools.getCPOSTag(pos);
            String lemma = token.lemma();
            Integer gov = indexedDeps.containsKey(idx) ? indexedDeps.get(idx).gov().index() : 0;
            String reln = indexedDeps.containsKey(idx) ? indexedDeps.get(idx).reln().toString() : "erased";
            String namedEntity = token.get(CoreAnnotations.NamedEntityTagAnnotation.class) == null ? "O" : token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
            DNode node = new DNode(idx, word, lemma, cPos, pos, reln);
            if (!namedEntity.equalsIgnoreCase("O")) {
                node.setNamedEntity(namedEntity);
            }

            node.addFeature("head", String.valueOf(gov));
            if (token.beginPosition() != -1) {
                node.addFeature("index_start", String.valueOf(token.beginPosition()));
            }

            if (token.endPosition() != -1) {
                node.addFeature("index_end", String.valueOf(token.endPosition()));
            }

            tree.add(node);
            idx++;
        }

        tree.stream().filter(x -> tree.getPaddingNode() != x).forEach(node -> {
            int headId = Integer.parseInt(node.getFeature("head"));
            node.setHead(tree.get(headId));
            tree.get(headId).addChild(node);
            node.getFeats().remove("head");
        });
        return tree;
    }

    private static void count(int counter, int size) {
        counter++;
        if (counter % 1000 == 0) {
            System.out.println("Processing " + counter + " of " + size);
        }
    }

    public static void convertTreebankToCoNLLX(String trainDirPath, FileFilter trainTreeBankFilter, String outputFileName) {
        DiskTreebank trainTreeBank = new DiskTreebank();
        trainTreeBank.loadPath(trainDirPath, trainTreeBankFilter);

        int counter = 0;
        int size = trainTreeBank.size();
        List<DTree> trees = trainTreeBank.parallelStream().map(tree -> {
            count(counter, size);
            return convertTreeBankToCoNLLX(tree.pennString());
        }).collect(Collectors.toList());

        try {
            FileWriter fw = new FileWriter(outputFileName);

            trees.forEach(dTree -> {
                try {
                    dTree.remove(0);
                    fw.write(dTree.toString());
                    fw.write(System.lineSeparator());
                    fw.write(System.lineSeparator());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            fw.flush();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Parser for tag Lemma
    public static DTree convertTreeBankToCoNLLX(final String constituentTree) {
        Tree tree = Tree.valueOf(constituentTree);

        SemanticHeadFinder headFinder = new SemanticHeadFinder(false); // keep copula verbs as head
        Collection<TypedDependency> dependencies = new EnglishGrammaticalStructure(tree, string -> true, headFinder).typedDependencies();
        List<CoreLabel> tokens = tree.taggedLabeledYield();
        StanfordParser.tagLemma(tokens);

        return StanfordTreeBuilder.generate(tokens, dependencies, null);
    }
}
