package org.maochen.datastructure;

import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (DNode node : this) {
            stringBuilder.append(node.toCoNLLString()).append(System.lineSeparator());
        }
        return stringBuilder.toString();
    }

    public List<DNode> getRoots() {
        return new ArrayList<>(Collections2.filter(this, new Predicate<DNode>() {
            @Override
            public boolean apply(DNode dNode) {
                return dNode.getDepLabel().equals("root");
            }
        }));
    }


    public DTree() {
        DNode padding = new DNode();
        padding.setId(0);
        padding.setName("^");
        this.add(padding);
    }
}
