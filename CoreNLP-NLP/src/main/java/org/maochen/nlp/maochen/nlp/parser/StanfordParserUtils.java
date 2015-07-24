package org.maochen.nlp.maochen.nlp.parser;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.*;
import org.maochen.nlp.maochen.nlp.datastructure.DTree;
import org.maochen.nlp.maochen.nlp.datastructure.LangTools;
import org.maochen.nlp.maochen.nlp.parser.stanford.StanfordParser;
import org.maochen.nlp.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.maochen.nlp.maochen.nlp.parser.stanford.pcfg.StanfordTreeBuilder;

import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordParserUtils {

    public static String getCoNLLXString(Collection<TypedDependency> deps, List<CoreLabel> tokens) {
        StringBuilder bf = new StringBuilder();

        Map<Integer, TypedDependency> indexedDeps = new HashMap<>(deps.size());
        for (TypedDependency dep : deps) {
            indexedDeps.put(dep.dep().index(), dep);
        }

        int idx = 1;

        if (tokens.get(0).lemma() == null) {
            StanfordNNDepParser.tagLemma(tokens);
        }

        for (CoreLabel token : tokens) {
            String word = token.word();
            String pos = token.tag();
            String cPos = (token.get(CoreAnnotations.CoarseTagAnnotation.class) != null) ?
                    token.get(CoreAnnotations.CoarseTagAnnotation.class) : LangTools.getCPOSTag(pos);
            String lemma = token.lemma();
            Integer gov = indexedDeps.containsKey(idx) ? indexedDeps.get(idx).gov().index() : 0;
            String reln = indexedDeps.containsKey(idx) ? indexedDeps.get(idx).reln().toString() : "erased";
            String out = String.format("%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t_\t_\n", idx, word, lemma, cPos, pos, gov, reln);
            bf.append(out);
            idx++;
        }
        bf.append("\n");
        return bf.toString();
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
                    fw.write(dTree.toString());
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
