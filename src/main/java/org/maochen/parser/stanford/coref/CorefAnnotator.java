package org.maochen.parser.stanford.coref;

import edu.stanford.nlp.dcoref.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Implements the Annotator for the new deterministic coreference resolution system.
 * In other words, this depends on: POSTaggerAnnotator, NERCombinerAnnotator (or equivalent), and ParserAnnotator.
 * <p>
 * Created by Maochen on 4/14/15.
 */
public class CorefAnnotator {


    private static final boolean VERBOSE = false;

    private final MentionExtractor mentionExtractor;
    private final SieveCoreferenceSystem corefSystem;

    private final boolean allowReparsing;

    public CorefAnnotator() {
        Properties props = new Properties();
        try {
            corefSystem = new SieveCoreferenceSystem(props);
            mentionExtractor = new MentionExtractor(corefSystem.dictionaries(), corefSystem.semantics());
            allowReparsing = PropertiesUtils.getBool(props, Constants.ALLOW_REPARSING_PROP, Constants.ALLOW_REPARSING);
        } catch (Exception e) {
            System.err.println("ERROR: cannot create CorefAnnotator!");
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public static String signature(Properties props) {
        return SieveCoreferenceSystem.signature(props);
    }

    public Map<Integer, CorefChain> annotate(List<CoreMap> annotatedSentences) {
        Map<Integer, CorefChain> result = null;
        List<Tree> trees = new ArrayList<Tree>();
        List<List<CoreLabel>> sentences = new ArrayList<List<CoreLabel>>();

        // extract trees and sentence words
        // we are only supporting the new annotation standard for this Annotator!
        boolean hasSpeakerAnnotations = false;
        for (CoreMap sentence : annotatedSentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            sentences.add(tokens);
            Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
            trees.add(tree);

            if (!hasSpeakerAnnotations) {
                // check for speaker annotations
                for (CoreLabel t : tokens) {
                    if (t.get(CoreAnnotations.SpeakerAnnotation.class) != null) {
                        hasSpeakerAnnotations = true;
                        break;
                    }
                }
            }
            MentionExtractor.mergeLabels(tree, tokens);
            MentionExtractor.initializeUtterance(tokens);
        }

        Annotation annotation = new Annotation(annotatedSentences);
        if (hasSpeakerAnnotations) {
            annotation.set(CoreAnnotations.UseMarkedDiscourseAnnotation.class, true);
        }

        // extract all possible mentions
        // this is created for each new annotation because it is not threadsafe
        RuleBasedCorefMentionFinder finder = new RuleBasedCorefMentionFinder(allowReparsing);
        List<List<Mention>> allUnprocessedMentions = finder.extractPredictedMentions(annotation, 0, corefSystem.dictionaries());

        try {
            // add the relevant info to mentions and order them for coref
            Document document = mentionExtractor.arrange(annotation, sentences, trees, allUnprocessedMentions);
            List<List<Mention>> orderedMentions = document.getOrderedMentions();
            if (VERBOSE) {
                for (int i = 0; i < orderedMentions.size(); i++) {
                    System.err.printf("Mentions in sentence #%d:\n", i);
                    for (int j = 0; j < orderedMentions.get(i).size(); j++) {
                        System.err.println("\tMention #" + j + ": " + orderedMentions.get(i).get(j).spanToString());
                    }
                }
            }

            result = corefSystem.coref(document);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

}
