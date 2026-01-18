from openai import OpenAI

def generate_answer(query, context_chunks, model="gpt-4o-mini", memory=None):
    client = OpenAI()

    ctx = "\n\n".join(context_chunks) if context_chunks else "No relevant excerpts found."

    messages = [{
        "role": "system",
        "content": (
            "You are bookfriend, a spoiler-aware lore assistant for the novel "
            "'Lord of the Mysteries'. Answer strictly using provided excerpts. "
            "If info is missing, say so. Avoid spoilers."
        )
    }]

    if memory:
        recent = memory.get_context(limit=6)
        if recent:
            messages.extend(recent)

    messages.append({
        "role": "user",
        "content": (
            f"User Query: {query}\n\n"
            f"Relevant Excerpts:\n{ctx}\n\n"
            "Answer strictly from the excerpts above. "
            "Mark any inference clearly."
        )
    })

    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6
    )

    return r.choices[0].message.content.strip()
