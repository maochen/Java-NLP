package org.maochen.parser;

import edu.stanford.nlp.io.NumberRangeFileFilter;
import edu.stanford.nlp.trees.*;
import org.maochen.datastructure.DTree;
import org.maochen.parser.stanford.pcfg.StanfordPCFGParser;
import org.maochen.utils.LangTools;

import java.io.FileFilter;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordParserUtils {

    public static void generateCoNLLXTree(String trainDirPath, int startRange, int endRange, boolean makeCopulaVerbHead) {
        DiskTreebank trainTreeBank = new DiskTreebank();
        FileFilter trainTreeBankFilter = new NumberRangeFileFilter(startRange, endRange, true);
        trainTreeBank.loadPath(trainDirPath, trainTreeBankFilter);

        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head
        trainTreeBank.stream().forEach(x -> new EnglishGrammaticalStructure(x, string -> true, headFinder, true));


        //                egs.buildCoNLLXGrammaticalStructure(List < List >)
    }

    public static void main(String[] args) {
        //        List<String> a = Lists.newArrayList("-treeFile", "/Users/Maochen/Desktop/tree.txt", "−basic", "−keepPunct", "−conllx");


//        StanfordNNDepParser depParser = new StanfordNNDepParser();

        String sentence = "Bill should went over the river and went through the woods.";
        sentence = "Mary can almost (always) tell when movies use fake dinosaurs and make changes.";
//                GrammaticalStructure egs = depParser.getGrammaticalStructure(sentence);
//                System.out.println(egs.typedDependenciesCollapsed());

        StanfordPCFGParser pcfgParser = new StanfordPCFGParser("", false);
        Tree tree = pcfgParser.getLexicalizedParser().parse(sentence);

        SemanticHeadFinder headFinder = new SemanticHeadFinder(false); // keep copula verbs as head
        // string -> true return all tokens including punctuations.
        GrammaticalStructure egs = new EnglishGrammaticalStructure(tree, string -> true, headFinder, true);

        System.out.println(egs.typedDependenciesCCprocessed());

        String conllx = EnglishGrammaticalStructure.dependenciesToString(egs, egs.typedDependenciesCCprocessed(), tree, true, true);
        System.out.println(conllx);
        DTree dtree = LangTools.getDTreeFromCoNLLXString(conllx, true);
//        System.out.println(dtree);
    }

}
