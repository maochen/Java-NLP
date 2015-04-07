package org.maochen.parser;

import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DNode;
import org.maochen.datastructure.LangLib;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 1/19/15.
 */
@Deprecated
public class StanfordTreeDirtyPatch {
    private static final Map<String, DNode> words = new HashMap<String, DNode>() {{
        DNode locate = new DNode(0, "located", "locate", LangLib.POS_VBD, StringUtils.EMPTY);
        put(locate.getLemma(), locate);
        DNode working = new DNode(0, "working", "work", LangLib.POS_VBG, StringUtils.EMPTY);
        put(working.getLemma(), working);
        DNode to = new DNode(1, "to", "to", LangLib.POS_IN, StringUtils.EMPTY);
        put(to.getLemma(), to);
        DNode in = new DNode(2, "in", "in", LangLib.POS_IN, StringUtils.EMPTY);
        put(in.getLemma(), in);
        DNode on = new DNode(2, "on", "on", LangLib.POS_IN, StringUtils.EMPTY);
        put(on.getLemma(), on);
        DNode blue = new DNode(2, "blue", "blue", LangLib.POS_JJ, StringUtils.EMPTY);
        put(blue.getLemma(), blue);
        DNode red = new DNode(2, "red", "red", LangLib.POS_JJ, StringUtils.EMPTY);
        put(red.getLemma(), red);
        DNode slow = new DNode(2, "slow", "slow", LangLib.POS_JJ, StringUtils.EMPTY);
        put(slow.getLemma(), slow);
        DNode french = new DNode(3, "french", "french", LangLib.POS_NNP, LangLib.DEP_NSUBJ);
        put(french.getLemma(), french);
        DNode insect = new DNode(4, "insect", "insect", LangLib.POS_NN, StringUtils.EMPTY);
        put(insect.getLemma(), insect);
        DNode username = new DNode(5, "username", "username", LangLib.POS_NN, StringUtils.EMPTY);
        put(username.getLemma(), username);
        DNode can = new DNode(6, "can", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(can.getLemma(), can);
        DNode could = new DNode(6, "could", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(could.getLemma(), could);
        DNode coulda = new DNode(6, "coulda", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(coulda.getLemma(), coulda);
        DNode shall = new DNode(6, "shall", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(shall.getLemma(), shall);
        DNode should = new DNode(6, "should", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(should.getLemma(), should);
        DNode shoulda = new DNode(6, "shoulda", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(shoulda.getLemma(), shoulda);
        DNode will = new DNode(6, "will", "will", LangLib.POS_MD, LangLib.DEP_AUX);
        put(will.getLemma(), will);
        DNode would = new DNode(6, "would", "will", LangLib.POS_MD, LangLib.DEP_AUX);
        put(would.getLemma(), would);
        DNode may = new DNode(6, "may", "may", LangLib.POS_MD, LangLib.DEP_AUX);
        put(may.getLemma(), may);
        DNode might = new DNode(6, "might", "may", LangLib.POS_MD, LangLib.DEP_AUX);
        put(might.getLemma(), might);
        DNode must = new DNode(6, "must", "must", LangLib.POS_MD, LangLib.DEP_AUX);
        put(must.getLemma(), must);
        DNode musta = new DNode(6, "musta", "must", LangLib.POS_MD, LangLib.DEP_AUX);
        put(musta.getLemma(), musta);
    }};


    private static final Map<String, String> auxVerbFix = new HashMap<String, String>() {
        {
            put("does", LangLib.POS_VBZ);
            put("did", LangLib.POS_VBD);
            put("do", LangLib.POS_VBP);
        }
    };

    // Most things here are stanford parsing issue.
    public static void dirtyPatch(DNode node) {
        // PS1: Don't fix the root to a verb if it is not. Ex: "a car." -> car is root.
        // PS2: "be simulated", actually the whole tree should start with node instead of "be", cannot fix the dep.

        // Dont assign verb that['s] to possessive
        if (node.getLemma().equalsIgnoreCase("'s") && !node.getPOS().startsWith(LangLib.POS_VB)) {
            node.setPOS(LangLib.POS_POS);
        }

        // Inconsistency in VBG and JJ
        // Ex: What is the reason for missing internal account? --> missing can be either JJ or VBG.
        else if (LangLib.POS_JJ.equals(node.getPOS()) && node.getLemma().endsWith("ing")) {
            node.setPOS(LangLib.POS_VBG);
        }

        // Mislabeled VBG as NN.
        else if (node.getPOS().startsWith(LangLib.POS_NN) && node.getLemma().endsWith("ing") && node.getDepLabel().equals(LangLib.DEP_ROOT)) {
            node.setPOS(LangLib.POS_VBG);
        }

        // If root has aux and itself is a Noun, correct it as verb.
        else if (node.getPOS().startsWith(LangLib.POS_NN) && node.getDepLabel().equals(LangLib.DEP_ROOT) && !node.getChildrenByDepLabels(LangLib.DEP_AUX).isEmpty()) {
            node.setPOS(LangLib.POS_VB);
        }

        // Fix root spread as verb
        else if (node.getLemma().toLowerCase().equals("spread") && node.getPOS().startsWith(LangLib.POS_NN) && node.getDepLabel().equals(LangLib.DEP_ROOT)) {
            node.setPOS(LangLib.POS_VBD);
        }

        DNode fixedNode = words.get(node.getLemma().toLowerCase());
        if (fixedNode != null) {
            if (node.getLemma().toLowerCase().equals("to")) {
                if (node.getDepLabel().equals(LangLib.DEP_PREP)) {
                    node.setPOS(LangLib.POS_IN);
                } else {
                    // Dont patch
                }
            }

            // French fix.
            else if ("french".equalsIgnoreCase(node.getLemma()) && node.getDepLabel().startsWith(LangLib.DEP_NSUBJ)) {
                node.setPOS(LangLib.POS_NNP);
            }

            // General Case
            else {
                if (fixedNode.getLemma() != null && !node.getLemma().equals(fixedNode.getLemma())) {
                    node.setLemma(fixedNode.getLemma());
                }

                if (fixedNode.getPOS() != null && !node.getPOS().equals(fixedNode.getPOS())) {
                    node.setPOS(fixedNode.getPOS());
                }

                if (fixedNode.getDepLabel() != null && !node.getDepLabel().equals(fixedNode.getDepLabel())) {
                    node.setDepLabel(fixedNode.getDepLabel());
                }
            }
        }

        // ---------- Fix the Label ------------
        // Fix the preposition label
        if (LangLib.POS_IN.equals(node.getPOS()) && !LangLib.DEP_MARK.equals(node.getDepLabel())) {
            node.setDepLabel(LangLib.DEP_PREP);
        }

        // For aux verb tagged as Noun.
        if (node.getId() == 1 && auxVerbFix.containsKey(node.getLemma().toLowerCase())) {
            node.setPOS(auxVerbFix.get(node.getLemma().toLowerCase()));
        }

        // hold together -> "together" should be PRT
        if (node.getLemma().equals("together") && node.getHead() != null && node.getHead().getLemma().equals("hold") && !node.getDepLabel().equals(LangLib.DEP_PRT)) {
            node.setDepLabel(LangLib.DEP_PRT);
        }

        // Ex: What bad weather.
        if (node.getPOS().equals(LangLib.POS_WDT) && node.getDepLabel().equals(LangLib.DEP_ATTR)) {
            node.setDepLabel(LangLib.DEP_DET);
        }
    }

    public static void dirtyPatchNER(DNode node) {
        if (!node.getNamedEntity().isEmpty()) {
            // 5pm. -> (. -> Time)
            if (node.getId() == node.getTree().size() - 1 && LangLib.DEP_PUNCT.equals(node.getDepLabel())) {
                node.setLemma(node.getForm());
                node.setNamedEntity(StringUtils.EMPTY);
            }

            // "Between XXXX", dont tag "Between"
            else if (LangLib.POS_IN.equals(node.getPOS()) && LangLib.NE_DATE.equalsIgnoreCase(node.getNamedEntity())) {
                node.setNamedEntity(StringUtils.EMPTY);
            }


            // Dirty Patch for Date.
            else if (node.getLemma().equalsIgnoreCase("and")) {
                if (node.getNamedEntity().equalsIgnoreCase(LangLib.NE_DATE) || node.getNamedEntity().equalsIgnoreCase(LangLib.NE_PERSON)) {
                    node.setNamedEntity(StringUtils.EMPTY);
                }
            }

            // Blame for stanford NER. Does Bill know John?  [ORG Does Bill]
            else if (node.getLemma().equalsIgnoreCase("does")) {
                node.setNamedEntity(StringUtils.EMPTY);
            }

        }
    }

}
