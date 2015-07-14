package org.maochen.utils;


import org.junit.Test;

import static org.junit.Assert.fail;

/**
 * Created by Maochen on 7/1/15.
 */
public class LangToolsTest {

    @Test
    public void testgetDTreeFromCoNLLXString() {
        String tree = "1\tThe\tthe\tDT\tDT\t_\t2\tdet\t_\t_\t_\n"
                + "2\trelative\trelative\tNN\tNN\t_\t3\tnsubj\t_\t_\t_\n"
                + "3\tpositions\tposition\tVBZ\tVBZ\tpb=position.01|vncls:9.1\t0\troot\t_\t_\t19:A1|6:A1\n"
                + "4\tto\tto\tTO\tTO\t_\t6\taux\t_\t_\t_\n"
                + "5\tbe\tbe\tVB\tVB\tvncls=109-1-1\t6\tauxpass\t_\t_\t_\n"
                + "6\tassumed\tassume\tVBN\tVBN\tpb=assume.02|vncls=93\t3\txcomp\t_\t_\t_\n"
                + "7\tby\tby\tIN\tIN\t_\t6\tprep\t_\t_\t6:A0\n"
                + "8\tmen\tman\tNNS\tNNS\t_\t7\tpobj\t_\t_\t_\n"
                + "9\tand\tand\tCC\tCC\t_\t8\tcc\t_\t_\t_\n"
                + "10\twomen\twoman\tNNS\tNNS\t_\t8\tconj\t_\t_\t_\n"
                + "11\tin\tin\tIN\tIN\t_\t6\tprep\t_\t_\t_\n"
                + "12\tthe\tthe\tDT\tDT\t_\t11\tpobj\t_\t_\t_\n"
                + "13\tworking\twork\tVBG\tVBG\tpb=work.01|vncls=73.1-3\t12\tamod\t_\t_\t_\n"
                + "14\tout\tout\tRP\tRP\t_\t12\tdep\t_\t_\t_\n"
                + "15\tof\tof\tIN\tIN\t_\t12\tprep\t_\t_\t_\n"
                + "16\tour\twe\tPRP$\tPRP$\t_\t17\tposs\t_\t_\t_\n"
                + "17\tcivilization\tcivilization\tNN\tNN\t_\t15\tpobj\t_\t_\t_\n"
                + "18\twere\tbe\tVBD\tVBD\tvncls=109-1-1\t19\tauxpass\t_\t_\t_\n"
                + "19\tassigned\tassign\tVBN\tVBN\tpb=assign.01|vncls=13.3-1\t17\trcmod\t_\t_\t_\n"
                + "20\tlong\tlong\tRB\tRB\t_\t21\tadvmod\t_\t_\t_\n"
                + "21\tago\tago\tRB\tRB\t_\t19\tadvmod\t_\t_\t19:AM-TMP\n"
                + "22\tby\tby\tIN\tIN\tvnrole=Agent\t19\tprep\t_\t_\t19:A0\n"
                + "23\ta\ta\tDT\tDT\t_\t25\tdet\t_\t_\t_\n"
                + "24\thigher\thigher\tJJR\tJJR\t_\t25\tamod\t_\t_\t_\n"
                + "25\tintelligence\tintelligence\tNN\tNN\t_\t22\tpobj\t_\t_\t_\n"
                + "26\t.\t.\t.\t.\t_\t3\tpunct\t_\t_\t_\n"
                + "27\t\"\t\"\t''''\t''''\t_\t3\tpunct\t_\t_\t_\n"
                + "28\t\t\tSYM\tSYM\tname=\t3\tdep\t_\t_\t_\n";

        try {
            LangTools.getDTreeFromCoNLLXString(tree);
        } catch (Exception e) {
            fail(e.toString());
        }
    }
}
