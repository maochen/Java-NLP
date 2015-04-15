package org.maochen.parser.stanford.coref;

import com.google.common.collect.Lists;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.lang3.StringUtils;
import org.maochen.parser.stanford.pcfg.StanfordPCFGParser;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 4/14/15.
 */
public class StanfordCoref {

    public StanfordPCFGParser parser = new StanfordPCFGParser();
    public CorefAnnotator corefAnnotator = new CorefAnnotator();

    public List<String> getCoref(List<String> texts) {
        List<CoreMap> sentences = texts.stream().map(parser::parseForCoref).collect(Collectors.toList());

        // Plural Coref gonna have multiple clusters ...
        Map<Integer, CorefChain> corefChainMap = corefAnnotator.annotate(sentences);
        for (Integer clusterID : corefChainMap.keySet()) {
            List<CorefChain.CorefMention> mentions = corefChainMap.get(clusterID).getMentionsInTextualOrder();
            if (mentions.size() < 2) {
                continue;
            }

            List<String> realEntities = mentions.stream().filter(x -> !x.mentionType.equals(Dictionaries.MentionType.PRONOMINAL)).map(x -> x.mentionSpan).collect(Collectors.toList());
            if (realEntities.isEmpty()) {
                continue;
            }
            mentions.stream().filter(x -> x.mentionType.equals(Dictionaries.MentionType.PRONOMINAL)).forEach(mention -> {
                List<CoreLabel> sentence = sentences.get(mention.sentNum - 1).get(CoreAnnotations.TokensAnnotation.class);
                // XXX: Handling plural coreferencing. "They", seems Stanford Plural doesn't work well. For example of the following, both of three entities are in 3 groups.
                // List<String> texts = Lists.newArrayList("Tom is nice.", "Mary is hard.", "They are all good.");

                // String replacedName = true ? realEntities.stream().reduce((s1, s2) -> s1 + ", " + s2).get() : realEntities.get(0);
                sentence.get(mention.startIndex - 1).setWord(realEntities.get(0));
                // Reset all possible trailing tokens.
                for (int i = mention.startIndex; i < mention.endIndex - 1; i++) {
                    sentence.get(i).setWord(StringUtils.EMPTY);
                }
            });
        }

        return sentences.stream().map(sentence -> sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(CoreLabel::word).reduce((w1, w2) -> w1 + StringUtils.SPACE + w2).get()).collect(Collectors.toList());
    }

    public static void main(String[] args) {
        StanfordCoref coref = new StanfordCoref();

        List<String> texts = Lists.newArrayList("Tom is nice.", "Mary is hard.", "They are all good.");
        coref.getCoref(texts).forEach(System.out::println);
    }
}
