package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

/**
 * Created by Maochen on 10/15/15.
 */
public class BinRelation extends TupleRelation {
    private Entity<?> left = null;
    private Entity<?> right = null;

    public Entity getLeft() {
        return left;
    }

    public void setLeft(Entity<?> left) {
        this.left = left;
        if (left != null) {
            left.relations.add(this);
        }
    }

    public Entity getRight() {
        return right;
    }

    public void setRight(Entity<?> right) {
        this.right = right;
        if (right != null) {
            right.relations.add(this);
        }
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("(")
                .append(super.getRel()).append(StringUtils.SPACE)
                .append("[").append(left).append("]").append(StringUtils.SPACE)
                .append("[").append(right).append("]")
                .append(") => ")
                .append(feats.values().stream().reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(StringUtils.EMPTY))
                .append(" - ").append(id);
        return stringBuilder.toString();
    }

//    public static void main(String[] args) {
//        BinRelation binRelation = new BinRelation();
//        binRelation.setRel("like");
//        binRelation.setRelType("VP");
//        binRelation.left = new Entity<>("Mary");
//        binRelation.right = new Entity<>("Tom");
//        System.out.println(binRelation);
//    }
}
