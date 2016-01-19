package org.maochen.nlp.parser.stanford;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Collections;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import edu.stanford.nlp.io.ExtensionFileFilter;
import edu.stanford.nlp.io.NumberRangeFileFilter;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.tagger.maxent.TaggerConfig;
import edu.stanford.nlp.trees.BobChrisTreeNormalizer;
import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeNormalizer;

/**
 * Created by Maochen on 4/20/15.
 */
public class StanfordPOSTaggerTrainer {
    public static final String wsj = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj/";
    public static final String extra = "/Users/Maochen/Desktop/extra/treebank_extra_data/";
    public static final String tempLocation = "/Users/Maochen/Desktop/tmp.txt";
    public static final String outputModelPath = "/Users/Maochen/Desktop/english-left3words-distsim.tagger";
    public static final String egw4_reut_512_clusters = StanfordPOSTaggerTrainer.class.getResource("/").getPath() + "/egw4-reut.512.clusters";

    private static void writeToFile(Set<String> data, String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            for (String str : data) {
                output.write(str);
                output.write(System.lineSeparator());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void loadTreeBank(FileFilter filter, String path, Collection<String> data) {
        DiskTreebank trainTreeBank = new DiskTreebank();
        trainTreeBank.loadPath(path, filter);

        final TreeNormalizer tn = new BobChrisTreeNormalizer();
        trainTreeBank.apply(treeVisitor -> {
            Tree tPrime = tn.normalizeWholeTree(treeVisitor, treeVisitor.treeFactory());
            data.add(Sentence.listToString(tPrime.taggedYield(), false, "_"));
        });
    }

    private static void convertTrainingData() {
        Set<String> trees = Collections.newSetFromMap(new ConcurrentHashMap<>());

        ForkJoinPool commonPool = ForkJoinPool.commonPool();

        Future task1 = commonPool.submit(() -> {
            FileFilter trainTreeBankFilter = new NumberRangeFileFilter(1, 2502, true);
            loadTreeBank(trainTreeBankFilter, wsj, trees);
            return null;
        });

        Future task2 = commonPool.submit(() -> {
            FileFilter extraTreeBankFilter = new ExtensionFileFilter(".mrg", true);
            loadTreeBank(extraTreeBankFilter, extra, trees);
            return null;
        });

        try {
            task2.get();
            task1.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        writeToFile(trees, tempLocation);
    }

    public static void main(String[] args) {

//        convertTrainingData();

        Properties props = new Properties();
        props.setProperty("mode", TaggerConfig.Mode.TRAIN.toString());
        props.setProperty("model", outputModelPath);
        props.setProperty("trainFile", "format=TSV,wordColumn=1,tagColumn=4," + tempLocation);
        props.setProperty("wordFunction", "edu.stanford.nlp.process.AmericanizeFunction");
        props.setProperty("closedClassTagThreshold", "40");
        props.setProperty("curWordMinFeatureThresh", "2");
        props.setProperty("encoding", "UTF-8");
        props.setProperty("iterations", "100");
        props.setProperty("lang", "english");
        props.setProperty("learnClosedClassTags", "false");
        props.setProperty("minFeatureThresh", "2");
        props.setProperty("rareWordMinFeatureThresh", "10");
        props.setProperty("rareWordThresh", "5");
//        props.setProperty("search", "cg");
        props.setProperty("sgml", "false");
        props.setProperty("sigmaSquared", "0.0");
        props.setProperty("regL1", "0.75");
        props.setProperty("tokenize", "true");
        props.setProperty("verbose", "false");
        props.setProperty("verboseResults", "true");
        props.setProperty("veryCommonWordThresh", "250");
        props.setProperty("outputFormat", "slashTags");
        props.setProperty("nthreads", "8");
        props.setProperty("tagSeparator", "\t");
        props.setProperty("arch", "left3words,naacl2003unknowns,wordshapes(-1,1),distsim(" + egw4_reut_512_clusters + ",-1,1),distsimconjunction(" + egw4_reut_512_clusters + ",-1,1)");

        TaggerConfig config = new TaggerConfig(props);


        try {
            Method m = MaxentTagger.class.getDeclaredMethod("trainAndSaveModel", TaggerConfig.class);
            m.setAccessible(true);
            m.invoke(null, config);
        } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

}
