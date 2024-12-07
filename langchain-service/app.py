from agent_service.hugging_face_agent_service import HuggingFaceAgentService

prompt = "Brandul de haine RaduClothingStore este un jucator important pe piata galateana inca din 2014. Poti face o postare te twitter care promoveaza acest brand?"

hugging_face_agent_service = HuggingFaceAgentService()
output = hugging_face_agent_service.agent_generate_content(prompt)

print(output)
