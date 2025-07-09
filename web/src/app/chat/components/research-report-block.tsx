// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import React from "react";
import { useCallback, useRef } from "react";
import { LoadingAnimation } from "~/components/deer-flow/loading-animation";
import { Markdown } from "~/components/deer-flow/markdown";
import ReportEditor from "~/components/editor";
import { useReplay } from "~/core/replay";
import { useMessage, useStore } from "~/core/store";
import { cn } from "~/lib/utils";

function extractTavilyBlocks(markdown: string) {
  // Match Tavily result blocks (badge at start, RAW_METADATA at end)
  const regex = /(🐦|💼|📸|👽|🌐)[\s\S]+?<!-- RAW_METADATA: ([^>]+) -->/g;
  let match;
  const blocks = [];
  while ((match = regex.exec(markdown))) {
    let meta = null;
    try {
      meta = JSON.parse(match[2]);
    } catch (e) {}
    blocks.push({
      raw: match[0],
      meta,
    });
  }
  return blocks;
}

export function ResearchReportBlock({
  className,
  messageId,
  editing,
}: {
  className?: string;
  researchId: string;
  messageId: string;
  editing: boolean;
}) {
  const message = useMessage(messageId);
  const { isReplay } = useReplay();
  const handleMarkdownChange = useCallback(
    (markdown: string) => {
      if (message) {
        message.content = markdown;
        useStore.setState({
          messages: new Map(useStore.getState().messages).set(
            message.id,
            message,
          ),
        });
      }
    },
    [message],
  );
  const contentRef = useRef<HTMLDivElement>(null);
  const isCompleted = message?.isStreaming === false && message?.content !== "";
  const content = message?.content || "";
  const tavilyBlocks = extractTavilyBlocks(content);

  return (
    <div ref={contentRef} className={cn("w-full pt-4 pb-8", className)}>
      {!isReplay && isCompleted && editing ? (
        <ReportEditor
          content={message?.content}
          onMarkdownChange={handleMarkdownChange}
        />
      ) : (
        <>
          {tavilyBlocks.length > 0 ? (
            <>
              {tavilyBlocks.map((block, i) => (
                <div key={i} className="rounded-lg border p-4 mb-4 bg-muted">
                  <div className="flex items-center gap-2 mb-2">
                    <span>{block.meta?.platform_emoji || "🌐"}</span>
                    <a
                      href={block.meta?.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-bold underline"
                    >
                      {block.meta?.title || block.meta?.url}
                    </a>
                  </div>
                  <div className="text-sm text-muted-foreground mb-2">
                    {block.meta?.author && (
                      <span className="mr-4">👤 {block.meta.author}</span>
                    )}
                    {block.meta?.timestamp && (
                      <span>🕒 {block.meta.timestamp}</span>
                    )}
                  </div>
                  {block.meta?.image_url && (
                    <img
                      src={block.meta.image_url}
                      alt="Preview"
                      className="w-full max-w-md rounded mb-2"
                    />
                  )}
                  <div className="mb-2">
                    <Markdown>{block.meta?.content || ""}</Markdown>
                  </div>
                  <div>
                    <a
                      href={block.meta?.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 underline"
                    >
                      🔗 Open in {block.meta?.platform || "source"}
                    </a>
                  </div>
                </div>
              ))}
              {/* Render the rest of the Markdown (non-Tavily content) */}
              <Markdown animated checkLinkCredibility>
                {content.replace(/(🐦|💼|📸|👽|🌐)[\s\S]+?<!-- RAW_METADATA: [^>]+ -->/g, "")}
              </Markdown>
            </>
          ) : (
            <>
              <Markdown animated checkLinkCredibility>
                {content}
              </Markdown>
              {message?.isStreaming && <LoadingAnimation className="my-12" />}
            </>
          )}
        </>
      )}
    </div>
  );
}
