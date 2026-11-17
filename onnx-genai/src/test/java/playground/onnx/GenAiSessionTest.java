package playground.onnx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Pure-Java parts only. The native path is exercised by actually running the chat. */
class GenAiSessionTest {

    @Test
    void jsonStringEscapesWhatChatMessagesContain() {
        assertEquals("\"plain\"", GenAiSession.jsonString("plain"));
        assertEquals("\"say \\\"hi\\\"\"", GenAiSession.jsonString("say \"hi\""));
        assertEquals("\"a\\\\b\"", GenAiSession.jsonString("a\\b"));
        assertEquals("\"line1\\nline2\\ttab\"", GenAiSession.jsonString("line1\nline2\ttab"));
        assertEquals("\"\\u0001\"", GenAiSession.jsonString("\u0001"));
        assertEquals("\"xin chào ☕\"", GenAiSession.jsonString("xin chào ☕"));
    }
}
