/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package classifier.backcup;

import opennlp.model.Event;

public class RealBasicEventStream {
    ListDataStream ds;
    Event next;

    public RealBasicEventStream(ListDataStream ds) {
        this.ds = ds;
        if (this.ds.hasNext()) next = createEvent((String) this.ds.nextToken());

    }

    public Event next() {
        while (next == null && this.ds.hasNext())
            next = createEvent((String) this.ds.nextToken());

        Event current = next;
        if (this.ds.hasNext()) {
            next = createEvent((String) this.ds.nextToken());
        }
        else {
            next = null;
        }
        return current;
    }

    public boolean hasNext() {
        while (next == null && ds.hasNext())
            next = createEvent((String) ds.nextToken());
        return next != null;
    }

    private Event createEvent(String obs) {
        int lastSpace = obs.lastIndexOf(' ');
        if (lastSpace == -1) return null;
        else {
            String[] contexts = obs.substring(0, lastSpace).split("\\s+");

            boolean hasRealValue = false;
            float[] values = new float[contexts.length];
            for (int ci = 0; ci < contexts.length; ci++) {
                int ei = contexts[ci].lastIndexOf("=");
                if (ei > 0 && ei + 1 < contexts[ci].length()) {
                    boolean gotReal = true;
                    try {
                        values[ci] = Float.parseFloat(contexts[ci].substring(ei + 1));
                    } catch (NumberFormatException e) {
                        gotReal = false;
                        System.err.println("Unable to determine value in context:" + contexts[ci]);
                        values[ci] = 1;
                    }
                    if (gotReal) {
                        if (values[ci] < 0) {
                            throw new RuntimeException("Negitive values are not allowed: " + contexts[ci]);
                        }
                        contexts[ci] = contexts[ci].substring(0, ei);
                        hasRealValue = true;
                    }
                }
                else {
                    values[ci] = 1;
                }
            }
            if (!hasRealValue) values = null;

            return new Event(obs.substring(lastSpace + 1), contexts, values);
        }
    }

}
