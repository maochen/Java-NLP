package org.maochen.nlp.sentencetypeclassifier;

import com.google.common.collect.Sets;

import org.maochen.nlp.datastructure.DNode;
import org.maochen.nlp.datastructure.DTree;
import org.maochen.nlp.datastructure.LangLib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by Maochen on 8/5/14.
 */
public class FeatureExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(FeatureExtractor.class);

    // Bossssssss.... currently all binary features.
    public List<String> generateFeats(String sentence, DTree tree) {
        List<String> feats = new ArrayList<>();

        feats.add("first_word_pos_" + tree.get(1).getPOS());

        DNode lastWord = tree.get(tree.size() - 1).getDepLabel().equals(LangLib.DEP_PUNCT) ? tree.get(tree.size() - 2) : tree.get(tree.size() - 1);
        feats.add("last_word_pos_" + lastWord.getPOS());

        if (tree.get(tree.size() - 1).getDepLabel().equals(LangLib.DEP_PUNCT)) {
            feats.add("punct_" + tree.get(tree.size() - 1).getLemma());
        }

        if (tree.getRoots().contains(tree.get(1))) {
            feats.add("first_word_root");
        }

        int auxCount = (int) tree.stream().parallel().filter(x -> LangLib.DEP_AUX.equals(x.getDepLabel())).distinct().count();
        if (auxCount > 0) {
            feats.add("has_aux_verb");
        }

        // Verify, Ask, Say - imperative
        Set<String> imperativeKeywords = Sets.newHashSet("verify", "ask", "say", "solve", "run", "execute");
        boolean isImperativeStart = imperativeKeywords.contains(tree.get(1).getLemma()) && tree.get(1).isRoot();
        if (isImperativeStart) {
            feats.add("imperative_start");
        }

        // whether
        DNode whether = tree.stream().filter(x -> "whether".equals(x.getLemma())).findFirst().orElse(null);
        if (whether != null) {
            feats.add("has_whether");
        }

        // Start with question words.
        Set<String> bagOfQuestionPrefix = Sets.newHashSet("tell me", "let me know", "clarify for me", "name");
        boolean isStartPrefixMatch = false;
        for (String prefix : bagOfQuestionPrefix) {
            if (sentence.toLowerCase().startsWith(prefix)) {
                isStartPrefixMatch = true;
                break;
            }
        }

        if (isStartPrefixMatch) {
            feats.add("question_bow_head");
        }

        List<String> biWord = new ArrayList<>();
        List<String> biDep = new ArrayList<>();

        List<String> triWord = new ArrayList<>();
        List<String> triDep = new ArrayList<>();

        for (int i = 1; i < tree.size(); i++) {
            DNode node = tree.get(i);
            if (tree.getPaddingNode() == node) {
                continue;
            }

            // Bigram
            if (i + 1 < tree.size()) {
                DNode nextNode = tree.get(i + 1);
                biWord.add(node.getForm().toLowerCase() + "_" + nextNode.getForm().toLowerCase());
                biDep.add(node.getDepLabel() + "_" + nextNode.getDepLabel());
            }

            // Trigram
            if (i + 2 < tree.size()) {
                DNode nextNode = tree.get(i + 1);
                DNode nextNextNode = tree.get(i + 2);
                triWord.add(node.getForm().toLowerCase() + "_" + nextNode.getForm().toLowerCase() + "_" + nextNextNode.getForm().toLowerCase());
                triDep.add(node.getDepLabel() + "_" + nextNode.getDepLabel() + "_" + nextNextNode.getDepLabel());
            }
        }

        feats.addAll(biWord);
        feats.addAll(biDep);
        feats.addAll(triWord);
        feats.addAll(triDep);

//        LOG.debug("feats: " + feats);
        return feats;
    }

}
