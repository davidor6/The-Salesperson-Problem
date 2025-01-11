export default function handler(req, res) {
  if (req.method === 'POST') {
    const { name } = req.body; // Extract data from the request body
    res.status(200).json({ message: `Node ${name} added successfully!` });
  } else {
    res.status(405).json({ error: 'Method Not Allowed' });
  }
}
