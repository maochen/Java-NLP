package org.maochen.parser;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DNode;
import org.maochen.datastructure.DTree;
import org.maochen.datastructure.LangLib;
import org.maochen.utils.LangTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * For the Stanford parse tree.
 * Created by Maochen on 10/28/14.
 */
public class StanfordTreeBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(StanfordTreeBuilder.class);

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

        // Take the 'dep' as 'attr'. Cuz ClearNLP says it is attr. funny right?
        if (node.getDepLabel().equals(LangLib.DEP_DEP)) {
            node.setDepLabel(LangLib.DEP_ATTR);
        }

        // ClearNLP does not have vmod.
        if (node.getDepLabel().equals(LangLib.DEP_VMOD)) {
            node.setDepLabel(LangLib.DEP_PARTMOD);
        }

        // This is due to the inconsistency of stanford parser.
        if (node.getDepLabel().equals(LangLib.DEP_NUMBER)) {
            node.setDepLabel(LangLib.DEP_NUM);
        }
    }

    // Most things here are stanford parsing issue.
    private static void dirtyPatch(DNode node) {
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
        if (node.getLemma().equals("together") && node.getParent() != null && node.getParent().getLemma().equals("hold") && !node.getDepLabel().equals(LangLib.DEP_PRT)) {
            node.setDepLabel(LangLib.DEP_PRT);
        }

        // Ex: What bad weather.
        if (node.getPOS().equals(LangLib.POS_WDT) && node.getDepLabel().equals(LangLib.DEP_ATTR)) {
            node.setDepLabel(LangLib.DEP_DET);
        }
    }

    private static void convertCopHead(DTree tree) {
        DNode originalRoot = tree.getRoots().get(0); // JJ mostly.
        List<DNode> cops = originalRoot.getChildrenByDepLabels(LangLib.DEP_COP);

        if (!originalRoot.getPOS().startsWith(LangLib.POS_VB) && !cops.isEmpty() && cops.get(0) != originalRoot) {
            DNode cop = cops.get(0);

            cop.setDepLabel(LangLib.DEP_ROOT);

            // Label might be corrected by dirty patch.
            if (originalRoot.getDepLabel().equals(LangLib.DEP_ROOT)) {
                originalRoot.setDepLabel(LangLib.DEP_DEP);
            }

            cop.setParent(tree.getPaddingNode());
            originalRoot.setParent(cop);

            tree.getPaddingNode().removeChild(originalRoot.getId());
            tree.getPaddingNode().addChild(cop);


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
            DNode node = new DNode(i + 1, token.originalText(), token.lemma(), token.tag(), StringUtils.EMPTY);
            // Set NamedEntity
            String namedEntity = getNamedEntity(token);
            if (!namedEntity.isEmpty()) {
                // Resolve Time
                if (namedEntity.equalsIgnoreCase(LangLib.NE_TIME)) {
                    String normalizedTime = token.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class);
                    if (normalizedTime != null) {
                        node.setName(normalizedTime);
                    } else {
                        LOG.warn("Time NamedEntity but doesn't has proper parsed time. " + token.originalText());
                    }
                }
                node.setNamedEntity(namedEntity);
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
                if (child == null) {
                    LOG.error(parent.getName() + " doesn't have proper child.");
                    continue;
                }
                // ClearNLP has different possessive handling.
                if (child.getPOS().equals(LangLib.POS_POS) && !childDEPLabel.equals(LangLib.DEP_POSSESSIVE)) {
                    childDEPLabel = LangLib.DEP_POSSESSIVE;
                    parent.setDepLabel(LangLib.DEP_POSS);
                }

                child.setDepLabel(childDEPLabel);
                setRel(parent, child);
            }
        }

        for (int i = 1; i < depTree.size(); i++) {
            DNode node = depTree.get(i);
            if (node.getDepLabel() == null) {
                if (node.getName().matches("\\p{Punct}+")) {
                    node.setDepLabel(LangLib.DEP_PUNCT);
                    setRel(depTree.get(rootVerbIndex), node);
                } else {
                    LOG.error("node does not have label. ->", node.toString());
                }
            }

            String namedEntity = node.getNamedEntity();
            if (!namedEntity.isEmpty()) {
                // 5pm. -> (. -> Time)
                if (node.getId() == depTree.size() - 1 && node.getDepLabel().equals(LangLib.DEP_PUNCT)) {
                    node.setLemma(node.getOriginalText());
                    node.setName(node.getOriginalText());
                    node.setNamedEntity(StringUtils.EMPTY);
                    continue;
                }

                int startNodeId = node.getId();
                for (; startNodeId >= 1; startNodeId--) {
                    String prevNE = depTree.get(startNodeId - 1).getNamedEntity().split("_")[0];
                    String currentNE = depTree.get(startNodeId).getNamedEntity().split("_")[0];
                    if (!prevNE.equals(currentNE)) {
                        break;
                    }
                }

                String formatted = namedEntity;
                formatted += startNodeId == node.getId() ? "_start_" : "_cont_";
                formatted += "${" + startNodeId + "}";

                node.setNamedEntity(formatted);
            }

            patchTree(node);
            dirtyPatch(node);
            LangTools.generateName(node);
        }
        // Dont put it before dirty patch.
        // Ex: Is the car slow? -> slow, VBZ should be correct to JJ first and then convert tree.
        convertCopHead(depTree);
        swapPossessives(depTree);
        return depTree;
    }

    private static void swapPossessives(DTree depTree) {

        Predicate<DNode> pred = (x) -> {
            if (x.getParent() == null) {
                return false;
            }
            boolean needAlter = x.getParent().isRoot();
            needAlter &= x.getPOS().startsWith(LangLib.POS_NN);
            return needAlter;
        };

        DNode originalParent = depTree.stream().parallel().filter(pred).findFirst().orElse(null);
        if (originalParent == null) {
            return;
        }


        DNode possessiveChild = originalParent.getChildren().stream().parallel().filter(x -> x.getLemma().equals("'s")).findFirst().orElse(null);
        if (possessiveChild == null) {
            return;
        }

        DNode nounChild = originalParent.getChildren().stream().parallel().filter(x -> x.getPOS().startsWith(LangLib.POS_NN)).findFirst().orElse(null);
        if (nounChild == null) {
            return;
        }

        DNode det = originalParent.getChildrenByDepLabels(LangLib.DEP_DET).stream().findFirst().orElse(null);
        if (det != null) {
            originalParent.removeChild(det.getId());
            nounChild.addChild(det);
            det.setParent(nounChild);
        }

        DNode originalGrandParent = originalParent.getParent();
        originalParent.removeChild(nounChild.getId());
        originalGrandParent.removeChild(originalParent.getId());

        // Swap DEP Labels.
        nounChild.setDepLabel(originalParent.getDepLabel());
        // Must be poss, dont use child's deplabel, it might be attr which is not accurate
        originalParent.setDepLabel(LangLib.DEP_POSS);

        nounChild.setParent(originalGrandParent);
        originalGrandParent.addChild(nounChild);

        originalParent.setParent(nounChild);
        nounChild.addChild(originalParent);
    }

    public static boolean isInvalidNE(final String token, final String currentNE) {
        // Dirty Patch for Date.
        if (token.equalsIgnoreCase("and")) {
            if (currentNE.equalsIgnoreCase(LangLib.NE_DATE) || currentNE.equalsIgnoreCase(LangLib.NE_PERSON)) {
                return true;
            }
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

        String type = token.ner();
        // Stanford doesn't need the next token patch rules there, just pass in empty.
        if (isInvalidNE(token.word(), type)) {
            return StringUtils.EMPTY;
        }

        // "Between XXXX", dont tag "Between"
        if (LangLib.POS_IN.equals(token.tag()) && LangLib.NE_DATE.equalsIgnoreCase(token.ner())) {
            return StringUtils.EMPTY;
        }

        return type;
    }

}
