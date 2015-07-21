package org.maochen.parser;

import org.maochen.datastructure.DTree;

/**
 * Created by Maochen on 12/8/14.
 */
public interface IParser {
    DTree parse(String sentence);
}
