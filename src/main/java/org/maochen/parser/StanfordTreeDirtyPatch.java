package org.maochen.parser;

import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DNode;
import org.maochen.datastructure.LangLib;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 1/19/15.
 */
public class StanfordTreeDirtyPatch {
    private static final Map<String, DNode> words = new HashMap<String, DNode>() {{
        DNode locate = new DNode(0, "located", "locate", LangLib.POS_VBD, StringUtils.EMPTY);
        put(locate.getName(), locate);
        DNode working = new DNode(0, "working", "work", LangLib.POS_VBG, StringUtils.EMPTY);
        put(working.getName(), working);
        DNode to = new DNode(1, "to", "to", LangLib.POS_IN, StringUtils.EMPTY);
        put(to.getName(), to);
        DNode in = new DNode(2, "in", "in", LangLib.POS_IN, StringUtils.EMPTY);
        put(in.getName(), in);
        DNode on = new DNode(2, "on", "on", LangLib.POS_IN, StringUtils.EMPTY);
        put(on.getName(), on);
        DNode blue = new DNode(2, "blue", "blue", LangLib.POS_JJ, StringUtils.EMPTY);
        put(blue.getName(), blue);
        DNode red = new DNode(2, "red", "red", LangLib.POS_JJ, StringUtils.EMPTY);
        put(red.getName(), red);
        DNode slow = new DNode(2, "slow", "slow", LangLib.POS_JJ, StringUtils.EMPTY);
        put(slow.getName(), slow);
        DNode french = new DNode(3, "french", "french", LangLib.POS_NNP, LangLib.DEP_NSUBJ);
        put(french.getName(), french);
        DNode insect = new DNode(4, "insect", "insect", LangLib.POS_NN, StringUtils.EMPTY);
        put(insect.getName(), insect);
        DNode username = new DNode(5, "username", "username", LangLib.POS_NN, StringUtils.EMPTY);
        put(username.getName(), username);
        DNode can = new DNode(6, "can", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(can.getName(), can);
        DNode could = new DNode(6, "could", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(could.getName(), could);
        DNode coulda = new DNode(6, "coulda", "can", LangLib.POS_MD, LangLib.DEP_AUX);
        put(coulda.getName(), coulda);
        DNode shall = new DNode(6, "shall", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(shall.getName(), shall);
        DNode should = new DNode(6, "should", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(should.getName(), should);
        DNode shoulda = new DNode(6, "shoulda", "shall", LangLib.POS_MD, LangLib.DEP_AUX);
        put(shoulda.getName(), shoulda);
        DNode will = new DNode(6, "will", "will", LangLib.POS_MD, LangLib.DEP_AUX);
        put(will.getName(), will);
        DNode would = new DNode(6, "would", "will", LangLib.POS_MD, LangLib.DEP_AUX);
        put(would.getName(), would);
        DNode may = new DNode(6, "may", "may", LangLib.POS_MD, LangLib.DEP_AUX);
        put(may.getName(), may);
        DNode might = new DNode(6, "might", "may", LangLib.POS_MD, LangLib.DEP_AUX);
        put(might.getName(), might);
        DNode must = new DNode(6, "must", "must", LangLib.POS_MD, LangLib.DEP_AUX);
        put(must.getName(), must);
        DNode musta = new DNode(6, "musta", "must", LangLib.POS_MD, LangLib.DEP_AUX);
        put(musta.getName(), musta);
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
        if (node.getName().equalsIgnoreCase("'s") && !node.getPOS().startsWith(LangLib.POS_VB)) {
            node.setPOS(LangLib.POS_POS);
        }

        // Inconsistency in VBG and JJ
        // Ex: What is the reason for missing internal account? --> missing can be either JJ or VBG.
        else if (LangLib.POS_JJ.equals(node.getPOS()) && node.getName().endsWith("ing")) {
            node.setPOS(LangLib.POS_VBG);
        }

        // Mislabeled VBG as NN.
        else if (node.getPOS().startsWith(LangLib.POS_NN) && node.getName().endsWith("ing") && node.getDepLabel().equals(LangLib.DEP_ROOT)) {
            node.setPOS(LangLib.POS_VBG);
        }

        // If root has aux and itself is a Noun, correct it as verb.
        else if (node.getPOS().startsWith(LangLib.POS_NN) && node.getDepLabel().equals(LangLib.DEP_ROOT) && !node.getChildrenByDepLabels(LangLib.DEP_AUX).isEmpty()) {
            node.setPOS(LangLib.POS_VB);
        }

        // Fix root spread as verb
        else if (node.getName().toLowerCase().equals("spread") && node.getPOS().startsWith(LangLib.POS_NN) && node.getDepLabel().equals(LangLib.DEP_ROOT)) {
            node.setPOS(LangLib.POS_VBD);
        }

        DNode fixedNode = words.get(node.getName().toLowerCase());
        if (fixedNode != null) {
            if (node.getName().toLowerCase().equals("to")) {
                if (node.getDepLabel().equals(LangLib.DEP_PREP)) {
                    node.setPOS(LangLib.POS_IN);
                } else {
                    // Dont patch
                }
            }

            // French fix.
            else if ("french".equalsIgnoreCase(node.getName()) && node.getDepLabel().startsWith(LangLib.DEP_NSUBJ)) {
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
        if (node.getId() == 1 && auxVerbFix.containsKey(node.getName().toLowerCase())) {
            node.setDepLabel(LangLib.DEP_AUX);
            node.setPOS(auxVerbFix.get(node.getName().toLowerCase()));
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
                node.setName(node.getForm());
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
