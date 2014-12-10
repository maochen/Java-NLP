package org.maochen.datastructure;

import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {

    private DNode padding;

    private String sentenceType = StringUtils.EMPTY;

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (DNode node : this) {
            if (node != padding) {
                stringBuilder.append(node.toString()).append(System.lineSeparator());
            }
        }
        return stringBuilder.toString();
    }

    public List<DNode> getRoots() {
        return new ArrayList<>(Collections2.filter(this, new Predicate<DNode>() {
            @Override
            public boolean apply(DNode dNode) {
                return dNode.getDepLabel().equals(LangLib.DEP_ROOT);
            }
        }));
    }

    public DNode getPaddingNode() {
        return padding;
    }

    public String getSentenceType() {
        return sentenceType;
    }

    public void setSentenceType(String sentenceType) {
        this.sentenceType = sentenceType;
    }

    public DTree() {
        padding = new DNode();
        padding.setId(0);
        padding.setName("^");
        this.add(padding);
    }
}
