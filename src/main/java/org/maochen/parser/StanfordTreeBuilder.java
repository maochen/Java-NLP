package org.maochen.parser;

import com.google.common.collect.ImmutableSet;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DNode;
import org.maochen.datastructure.DTree;
import org.maochen.datastructure.LangLib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by Maochen on 12/8/14.
 */
public class StanfordTreeBuilder {

    private static final Logger LOG = LoggerFactory.getLogger(StanfordTreeBuilder.class);

    private static final Set<String> preps = ImmutableSet.of("to", "in", "on");
    private static final Set<String> modalVerbs = ImmutableSet.of("can", "could", "coulda", "shall", "should", "shoulda", "will", "would", "may", "might", "must", "musta");
    private static final Set<String> adjectives = ImmutableSet.of("blue", "red", "slow");
    private static final Set<String> nouns = ImmutableSet.of("insect", "username");

    private static final Map<String, String> auxVerbFix = new HashMap<String, String>() {
        {
            put("does", LangLib.POS_VBZ);
            put("did", LangLib.POS_VBD);
            put("do", LangLib.POS_VBP);
        }
    };

    // if pos is IN, and dep label is the following
    private static final Set<String> excludingPrepDepLabels = ImmutableSet.of("mark");

    private static final String NAMEDENTITY = "name_entity";

    private static void setRel(DNode parent, DNode child) {
        child.setParent(parent);
        parent.addChild(child);
    }

    // This is because of the difference of the stanford vs clearnlp standards. It is not err.
    private static void patchTree(DNode node) {
        if (node.getName().equalsIgnoreCase("-LRB-")) {
            node.setName("(");
        } else if (node.getName().equalsIgnoreCase("-RRB-")) {
            node.setName(")");
        }

        if (preps.contains(node.getName().toLowerCase())) {
            if (node.getName().toLowerCase().equals("to") && node.getDepLabel().equals("prep")) {
                node.setPos(LangLib.POS_IN);
            } else if (!node.getName().toLowerCase().equals("to")) {
                node.setPos(LangLib.POS_IN);
            }
        }

        // Take the 'dep' as 'attr'. Cuz ClearNLP says it is attr. funny right?
        if (node.getDepLabel().equals(LangLib.DEP_DEP)) {
            node.setDepLabel(LangLib.DEP_ATTR);
        }

        // ClearNLP does not have vmod.
        if (node.getDepLabel().equals("vmod")) {
            node.setDepLabel(LangLib.DEP_PARTMOD);
        }
    }

    // TODO: Most things here are stanford parsing issue. Try to get rid of these once we can train our own stanford models.
    private static void dirtyPatch(DNode node) {
        // ---------- Fix the POS ---------------
        // Don't fix the root to a verb if it is not. Ex: "a car." -> car is root.

        // Dont assign verb that['s] to possessive
        if (node.getName().equalsIgnoreCase("'s") && !node.getPos().startsWith(LangLib.POS_VB)) {
            node.setPos(LangLib.POS_POS);
        }

        // TODO: we should put this into the interrogativeAttributeMatcherDelegate to match JJ and VBG instead of correct POS here.
        // Inconsistency in VBG and JJ
        // Ex: What is the reason for missing internal account? --> missing can be either JJ or VBG.
        else if (LangLib.POS_JJ.equals(node.getPos()) && node.getName().endsWith("ing")) {
            node.setPos(LangLib.POS_VBG);
        }

        // adjective fix.
        else if (adjectives.contains(node.getName().toLowerCase())) {
            node.setPos(LangLib.POS_JJ);
        }

        // noun fix.
        else if (nouns.contains(node.getName().toLowerCase())) {
            node.setPos(LangLib.POS_NN);
        }

        // French fix.
        else if ("french".equalsIgnoreCase(node.getName()) && node.getDepLabel().startsWith(LangLib.DEP_NSUBJ)) {
            node.setPos(LangLib.POS_NNP);
        }

        // ---------- Fix the Label ------------
        // Fix the preposition label
        if (LangLib.POS_IN.equals(node.getPos()) && !excludingPrepDepLabels.contains(node.getDepLabel())) {
            node.setDepLabel(LangLib.DEP_PREP);
        }

        // For aux verb tagged as Noun.
        if (node.getId() == 1 && auxVerbFix.containsKey(node.getName().toLowerCase())) {
            node.setDepLabel(LangLib.DEP_AUX);
            node.setPos(auxVerbFix.get(node.getName().toLowerCase()));
        }

        // Modal fix.
        if (modalVerbs.contains(node.getName().toLowerCase())) {
            node.setPos(LangLib.POS_MD);
            node.setDepLabel(LangLib.DEP_AUX);
        }

        // hold together -> "together" should be PRT
        if (node.getName().equalsIgnoreCase("together") && node.getParent() != null && node.getParent().getLemma().equals("hold") && !node.getDepLabel().equals(LangLib.DEP_PRT)) {
            node.setDepLabel(LangLib.DEP_PRT);
        }
    }

    private static void convertCopHead(DTree tree) {
        DNode originalRoot = tree.getRoots().get(0); // JJ mostly.
        List<DNode> cops = originalRoot.getChildrenByDepLabels("cop");

        if (!originalRoot.getPos().startsWith(LangLib.POS_VB) && !cops.isEmpty() && cops.get(0) != originalRoot) {
            DNode headOfRoot = tree.get(0);
            DNode cop = cops.get(0);

            cop.setDepLabel("root");
            originalRoot.setDepLabel("dep");

            cop.setParent(headOfRoot);
            originalRoot.setParent(cop);

            headOfRoot.removeChild(originalRoot.getId());
            headOfRoot.addChild(cop);

            // Add original deps to cop
            for (DNode child : originalRoot.getChildren()) {
                originalRoot.removeChild(child.getId());

                if (child != cop) {
                    cop.addChild(child);
                    child.setParent(cop);
                }
            }

            cop.addChild(originalRoot);
        }
    }

    public static DTree generate(List<CoreLabel> tokens, Tree tree, Collection<TypedDependency> dependencies) {
        tree.setSpans();
        DTree depTree = new DTree();
        for (int i = 0; i < tokens.size(); i++) {
            CoreLabel token = tokens.get(i);
            DNode node = new DNode();
            node.setId(i + 1);
            node.setName(token.originalText());
            node.setLemma(token.lemma());
            node.setPos(token.tag());

            // Set NamedEntity
            String namedEntity = getNamedEntity(token);
            if (!namedEntity.isEmpty()) {
                if (namedEntity.equalsIgnoreCase(LangLib.NE_TIME)) {
                    String normalizedTime = token.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class);
                    if (normalizedTime != null) {
                        node.setLemma(normalizedTime);
                    } else {
                        LOG.warn("Time NamedEntity but doesn't has proper parsed time. " + token.originalText());
                    }
                }
                node.addFeature(NAMEDENTITY, namedEntity);
            }

            depTree.add(node);
        }

        // 0 is _R_ here
        int rootVerbIndex = -1;
        for (TypedDependency td : dependencies) {
            int sourceIndex = td.gov().index();
            int targetIndex = td.dep().index();
            String childDEPLabel = td.reln().toString();


            if (sourceIndex == 0) {
                rootVerbIndex = targetIndex;
            }

            if (sourceIndex != -1) {
                DNode child = depTree.get(targetIndex);
                DNode parent = depTree.get(sourceIndex);
                // ClearNLP has different possessive handling.
                if (child.getPos().equals(LangLib.POS_POS) && !childDEPLabel.equals(LangLib.DEP_POSSESSIVE)) {
                    childDEPLabel = LangLib.DEP_POSSESSIVE;
                    parent.setDepLabel(LangLib.DEP_POSS);
                }

                child.setDepLabel(childDEPLabel);
                setRel(parent, child);
            }
        }


        for (int i = 1; i < depTree.size(); i++) {
            DNode node = depTree.get(i);
            if (node.getDepLabel().isEmpty()) {
                if (node.getName().matches("\\p{Punct}+")) {
                    node.setDepLabel(LangLib.DEP_PUNCT);
                    setRel(depTree.get(rootVerbIndex), node);
                } else {
                    LOG.error("node does not have label. ->", node.toString());
                }
            }

            String namedEntity = node.getFeature(NAMEDENTITY);
            if (namedEntity != null) {
                // 5pm. -> (. -> Time)
                if (node.getId() == depTree.size() - 1 && node.getDepLabel().equals(LangLib.DEP_PUNCT)) {
                    node.setLemma(node.getName());
                    node.removeFeature(NAMEDENTITY);
                    continue;
                }

                int startNodeId = node.getId();
                for (; startNodeId >= 1; startNodeId--) {
                    DNode prev = depTree.get(startNodeId - 1);
                    DNode current = depTree.get(startNodeId);
                    if (!getNamedEntity(current).equals(getNamedEntity(prev))) {
                        break;
                    }
                }

                String formatted = namedEntity;
                formatted += startNodeId == node.getId() ? "_start_" : "_cont_";
                formatted += "${" + startNodeId + "}";

                node.addFeature(NAMEDENTITY, formatted);
            }

            patchTree(node);
            dirtyPatch(node);
        }

        convertCopHead(depTree);
        return depTree;
    }


    private static String getNamedEntity(DNode node) {
        boolean noNamedEntity = node == null;
        noNamedEntity |= node.getFeature(NAMEDENTITY) == null;
        noNamedEntity |= node.getPos().startsWith(LangLib.POS_VB);
        // 5pm is JJ and Time
        // noNamedEntity |= node.pos.startsWith(CTLibEn.POS_JJ);

        return noNamedEntity ? StringUtils.EMPTY : node.getFeature(NAMEDENTITY).split("_")[0];
    }

    public static boolean isValidNamedEntity(final String token, final String currentNE) {
        // Dirty Patch for Date.
        if (token.equalsIgnoreCase("and")) {
            if (currentNE.equalsIgnoreCase(LangLib.NE_DATE) || currentNE.equalsIgnoreCase(LangLib.NE_PERSON)) {
                return true;
            }
        }
        // Blame for the openNLP NER, it tags $5000 to DATE!!
        else if (token.startsWith("$") && currentNE.equalsIgnoreCase(LangLib.NE_DATE)) {
            return true;
        }
        // Blame for the openNLP NER again, it tags ',' inside person [PERSON John, Mary]
        else if (token.equals(",") && currentNE.equalsIgnoreCase(LangLib.NE_PERSON)) {
            return true;
        }

        // Blame for stanford NER. Does Bill know John?  [ORG Does Bill]
        else if (token.equalsIgnoreCase("does")) {
            return true;
        }

        return false;
    }

    private static String getNamedEntity(CoreLabel token) {
        if (token.ner() == null || token.ner().equals("O")) {
            return StringUtils.EMPTY;
        }

        String type = token.ner().toUpperCase();
        // Stanford doesn't need the next token patch rules there, just pass in empty.
        if (isValidNamedEntity(token.word(), type)) {
            return StringUtils.EMPTY;
        }

        // "Between XXXX", dont tag "Between"
        if (LangLib.POS_IN.equals(token.tag()) && LangLib.NE_DATE.equalsIgnoreCase(token.ner())) {
            return StringUtils.EMPTY;
        }


        return type;
    }


}
