package org.maochen.parser;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.NumberRangeFileFilter;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.sequences.DocumentReaderAndWriter;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;

/**
 * Created by Maochen on 1/9/15.
 */
public class StanfordParserTrainer {
    public static final String desktop = "/Users/Maochen/Desktop/";
    public static final String modelPath = desktop + "/englishPCFG.ser.gz";
    public static final String wsj = desktop + "treebank_3/parsed/mrg/wsj";

    public static void trainPCFG(String wsjPath, String modelPath, int maxLength, int nthreads) {
        File f = new File(modelPath);
        if (f.exists()) {
            System.out.println("Delete the existing model in " + f.getAbsolutePath());
            f.delete();
        }


        String para = "-goodPCFG -maxLength " + maxLength + " -nthreads " + nthreads;

        Options op = new Options();
        op.setOptions(para.split("\\s"));

        DiskTreebank extraTreeBank = op.tlpParams.diskTreebank();
        FileFilter extraTreeBankFilter = new NumberRangeFileFilter(2500, 2501, true);
        extraTreeBank.loadPath(wsjPath, extraTreeBankFilter);


        DiskTreebank wsjTreeBank = op.tlpParams.diskTreebank();
        FileFilter wsjTreeBankFilter = new NumberRangeFileFilter(1, 2454, true);
        wsjTreeBank.loadPath(wsjPath, wsjTreeBankFilter);

        LexicalizedParser.getParserFromTreebank(extraTreeBank, wsjTreeBank, 0.01, null, op, null, null).saveParserToSerialized(modelPath);
    }

    public static void trainNER() {
        String para = "-prop" + desktop + "ner.prop";
        try {
            CRFClassifier.main(para.split("\\s"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void ner(String[] args, String serializeTo) throws Exception {
        Properties props = StringUtils.argsToProperties(args);
        SeqClassifierFlags flags = new SeqClassifierFlags(props);

        Method m = CRFClassifier.class.getDeclaredMethod("chooseCRFClassifier", SeqClassifierFlags.class);
        m.setAccessible(true); //if security settings allow this
        CRFClassifier<CoreLabel> crf = (CRFClassifier<CoreLabel>) m.invoke(null, flags);

        String testFile = flags.testFile; //null
        String testFiles = flags.testFiles; //null
        String textFile = flags.textFile;
        String textFiles = flags.textFiles;
        String loadPath = flags.loadClassifier;
        String loadTextPath = flags.loadTextClassifier;
        String serializeToText = flags.serializeToText;
        int vectorSize;
        int var25;
        if (crf.flags.useEmbedding && crf.flags.embeddingWords != null && crf.flags.embeddingVectors != null) {
            System.err.println("Reading Embedding Files");
            BufferedReader files = IOUtils.readerFromString(crf.flags.embeddingWords);
            List<String> dict = new ArrayList<>();

            String count;
            while ((count = files.readLine()) != null) {
                dict.add(count.trim());
            }

            System.err.println("Found a dictionary of size " + dict.size());
            files.close();

            // Just binding the embeddings var.
            Map<String, double[]> embeddings = new HashMap<>();
            Field embeddingsField = CRFClassifier.class.getDeclaredField("embeddings");
            embeddingsField.setAccessible(true);
            embeddingsField.set(crf, embeddings);
            //------------

            var25 = 0;
            vectorSize = -1;
            boolean filename = false;

            String line;
            double[] vector;
            for (files = IOUtils.readerFromString(crf.flags.embeddingVectors); (line = files.readLine()) != null; embeddings.put(dict.get(var25++), vector)) {
                vector = ArrayUtils.toDoubleArray(line.trim().split(" "));
                if (vectorSize < 0) {
                    vectorSize = vector.length;
                } else if (vectorSize != vector.length && !filename) {
                    System.err.println("Inconsistent vector lengths: " + vectorSize + " vs. " + vector.length);
                    filename = true;
                }
            }

            System.err.println("Found " + var25 + " matching embeddings of dimension " + vectorSize);
        }

        if (crf.flags.loadClassIndexFrom != null) {
            crf.classIndex = CRFClassifier.loadClassIndexFromFile(crf.flags.loadClassIndexFrom);
        }

        if (loadPath != null) {
            crf.loadClassifierNoExceptions(loadPath, props);
        } else if (loadTextPath != null) {
            System.err.println("Warning: this is now only tested for Chinese Segmenter");
            System.err.println("(Sun Dec 23 00:59:39 2007) (pichuan)");

            try {
                crf.loadTextClassifier(loadTextPath, props);
            } catch (Exception var19) {
                throw new RuntimeException("error loading " + loadTextPath, var19);
            }
        } else if (crf.flags.loadJarClassifier != null) {
            crf.loadJarClassifier(crf.flags.loadJarClassifier, props);
        } else if (crf.flags.trainFile == null && crf.flags.trainFileList == null) {
            crf.loadDefaultClassifier();
        } else {
            Timing var20 = new Timing();
            crf.train();
            var20.done("CRFClassifier training");
        }

        crf.loadTagIndex();
        if (serializeTo != null) {
            crf.serializeClassifier(serializeTo);
        }

        if (crf.flags.serializeWeightsTo != null) {
            crf.serializeWeights(crf.flags.serializeWeightsTo);
        }

        if (crf.flags.serializeFeatureIndexTo != null) {
            crf.serializeFeatureIndex(crf.flags.serializeFeatureIndexTo);
        }

        if (serializeToText != null) {
            crf.serializeTextClassifier(serializeToText);
        }

        if (testFile != null) {
            DocumentReaderAndWriter<CoreLabel> var22 = crf.defaultReaderAndWriter();
            if (crf.flags.searchGraphPrefix != null) {
                crf.classifyAndWriteViterbiSearchGraph(testFile, crf.flags.searchGraphPrefix, crf.makeReaderAndWriter());
            } else if (crf.flags.printFirstOrderProbs) {
                crf.printFirstOrderProbs(testFile, var22);
            } else if (crf.flags.printFactorTable) {
                crf.printFactorTable(testFile, var22);
            } else if (crf.flags.printProbs) {
                crf.printProbs(testFile, var22);
            } else if (crf.flags.useKBest) {
                int var24 = crf.flags.kBest;
                crf.classifyAndWriteAnswersKBest(testFile, var24, var22);
            } else if (crf.flags.printLabelValue) {
                crf.printLabelInformation(testFile, var22);
            } else {
                crf.classifyAndWriteAnswers(testFile, var22, true);
            }
        }

        String[] var21;
        List<File> var23 = new ArrayList<>();
        String var26;
        if (testFiles != null) {
            var21 = testFiles.split(",");
            var25 = var21.length;

            for (vectorSize = 0; vectorSize < var25; ++vectorSize) {
                var26 = var21[vectorSize];
                var23.add(new File(var26));
            }

            crf.classifyFilesAndWriteAnswers(var23, crf.defaultReaderAndWriter(), true);
        }

        if (textFile != null) {
            crf.classifyAndWriteAnswers(textFile);
        }

        if (textFiles != null) {
            var21 = textFiles.split(",");
            var25 = var21.length;

            for (vectorSize = 0; vectorSize < var25; ++vectorSize) {
                var26 = var21[vectorSize];
                var23.add(new File(var26));
            }

            crf.classifyFilesAndWriteAnswers(var23);
        }

        if (crf.flags.readStdin) {
            crf.classifyStdin();
        }

    }

    public static void printParseTree(LexicalizedParser parser, String sentence) {
        // Parse right after get through tokenizer.
        Tree tree = parser.parse(sentence);
        System.out.println(tree.pennString());

        SemanticHeadFinder headFinder = new SemanticHeadFinder(false); // keep copula verbs as head
        Collection<TypedDependency> dependencies = new EnglishGrammaticalStructure(tree, string -> true, headFinder).typedDependencies();
        dependencies.stream().forEach(System.out::println);
    }

    public static void main(String[] args) {
        //        trainNER();

        trainPCFG(wsj, modelPath, 80, 6);

        StanfordParser parser = new StanfordParser();

        parser.loadModel("/Users/Maochen/Desktop/englishPCFG.ser.gz");

        //        parser.getLexicalizedParser().saveParserToTextFile(desktop + "txt.txt");
        Scanner scan = new Scanner(System.in);
        String input = "";
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
                printParseTree(parser.getLexicalizedParser(), input);
            }
        }


    }

}
