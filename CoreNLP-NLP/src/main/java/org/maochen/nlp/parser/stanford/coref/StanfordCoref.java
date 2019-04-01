//package org.maochen.nlp.parser.stanford.coref;
//
//import org.apache.commons.lang3.StringUtils;
//
//import java.io.IOException;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.Properties;
//import java.util.stream.Collectors;
//
//import edu.stanford.nlp.coref.CorefAlgorithm;
//import edu.stanford.nlp.coref.CorefCoreAnnotations;
//import edu.stanford.nlp.coref.data.Dictionaries;
//import edu.stanford.nlp.coref.data.Document;
//import edu.stanford.nlp.coref.data.DocumentMaker;
//import edu.stanford.nlp.coref.data.InputDoc;
//import edu.stanford.nlp.coref.data.Mention;
//import edu.stanford.nlp.coref.neural.NeuralCorefAlgorithm;
//import edu.stanford.nlp.ling.CoreAnnotations;
//import edu.stanford.nlp.ling.CoreLabel;
//import edu.stanford.nlp.pipeline.Annotation;
//import edu.stanford.nlp.pipeline.StanfordCoreNLP;
//
///**
// * Created by Maochen on 4/14/15.
// */
//public class StanfordCoref {
//
//    private final CorefAlgorithm corefAlgorithm;
//    private final Dictionaries dictionaries;
//
//    private Properties props;
////    private final StanfordNNDepParser parser;
//
//    public StanfordCoref() {
//        try {
//            props = new Properties();
//            dictionaries = new Dictionaries();
//            corefAlgorithm = new NeuralCorefAlgorithm(props, dictionaries);
//        } catch (ClassNotFoundException | IOException e) {
//            e.printStackTrace();
//            throw new RuntimeException(e);
//        }
//
//    }
//
//    public List<String> getCoref(List<String> texts) {
//        Properties props = new Properties();
//        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, mention");
//        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//
//        String text = texts.stream().collect(Collectors.joining(StringUtils.SPACE));
//        Annotation annotation = new Annotation(text);
//        pipeline.annotate(annotation);
//
//        InputDoc inputDoc = new InputDoc(annotation);
//        Document document;
//        try {
//            DocumentMaker documentMaker = new DocumentMaker(props, new Dictionaries());
//            document = documentMaker.makeDocument(inputDoc);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//
//        corefAlgorithm.runCoref(document);
//
//        Map<Mention, Mention> pronToNoun = new HashMap<>();
//        document.corefClusters.values().forEach(x -> {
//
//            x.corefMentions.forEach(mentionEntity -> {
//                if (mentionEntity.mentionType == Dictionaries.MentionType.PRONOMINAL) {
//                    pronToNoun.put(mentionEntity, x.representative);
//                }
//            });
//        });
//
//
//        return document.annotation.get(CoreAnnotations.SentencesAnnotation.class).stream().map(singleSentence -> {
//            List<Mention> mentions = singleSentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class);
//            Map<CoreLabel, String> tokenToWord = new HashMap<>();
//            for (Mention mention : mentions) {
//                if (pronToNoun.containsKey(mention)) {
//                    tokenToWord.put(mention.headIndexedWord.backingLabel(), pronToNoun.get(mention).headString);
//                }
//            }
//
//            return singleSentence.get(CoreAnnotations.TokensAnnotation.class).stream()
//                    .map(token -> {
//                        if (tokenToWord.containsKey(token)) {
//                            String word = tokenToWord.get(token);
//
//                            if (token.get(CoreAnnotations.IndexAnnotation.class) == 1) {
//                                word = StringUtils.capitalize(word);
//                            }
//
//                            return word;
//                        } else {
//                            return token.get(CoreAnnotations.TextAnnotation.class);
//                        }
//                    })
//                    .collect(Collectors.joining(StringUtils.SPACE));
//        }).collect(Collectors.toList());
//    }
//}
