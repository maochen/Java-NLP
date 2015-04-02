package org.maochen.utils;

import org.maochen.datastructure.DNode;
import org.maochen.datastructure.LangLib;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 12/10/14.
 */
public class LangTools {

    /**
     * http://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
     */
    private static final Map<String, String> contractions = new HashMap<String, String>() {{
        put("'m", "am");
        put("'re", "are");
        put("'ve", "have");
        put("can't", "cannot");
        put("ma'am", "madam");
        put("'ll", "will");
    }};

    public static void generateName(DNode node) {
        // Can't
        if (node.getForm().equalsIgnoreCase("ca") && node.getLemma().equals("can")) {
            node.setName(node.getLemma());
        }
        // I ca/[n't].
        else if (node.getForm().equalsIgnoreCase("n't") && node.getLemma().equals("not") && node.getDepLabel().equals(LangLib.DEP_NEG)) {
            node.setName(node.getLemma());

        }

        // Resolve 'd
        else if (node.getForm().equalsIgnoreCase("'d") && node.getPOS().equals(LangLib.POS_MD)) {
            node.setName(node.getLemma());
        } else if (contractions.containsKey(node.getForm())) {
            node.setName(contractions.get(node.getForm()));
        }
    }
}
